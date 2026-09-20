"""
A2A v1.0 Agent Card signature verification — JWKS + detached JWS.

Implements the signature side of the A2A v1.0 Agent Card spec:

  * Cards carry detached-JWS (RFC 7515) signatures in ``signatures[]``; the
    signed payload is the RFC 8785-canonicalized card JSON with the
    ``signatures`` field omitted.
  * Public keys are provided as a JWKS (``{"keys": [...]}``) document, e.g.
    the card producer's ``/.well-known/jwks.json``.

This module requires the optional ``a2a`` extras (``pyjwt`` +
``cryptography``) and imports them lazily, so the core AgentLink package
keeps zero new hard dependencies::

    pip install "cdzzy-agentlink[a2a]"

Canonicalization note: full RFC 8785 (JCS) canonicalization is not implemented
by the standard library; we use a deterministic simplification (sorted keys,
compact separators, UTF-8) that is byte-stable across processes and platforms.
``sign_agent_card`` and ``verify_agent_card`` share it, so signatures produced
by AgentLink always verify with AgentLink. Cards signed by other JCS
implementations verify as long as their canonicalization matches (true for
cards without floating-point numbers).

Typical flow (self-signed dev card):

    private_pem, jwks = generate_signing_key("ES256", kid="dev-key")
    card = AgentCard(name="my-agent", ...)
    card = sign_agent_card(card, private_pem, kid="dev-key")
    verify_agent_card(card, jwks)   # -> True, raises on tampering
"""

from __future__ import annotations

import base64
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from agentlink.a2a.card import AgentCard, AgentCardSignature

# JWS algorithms supported for Agent Card signatures.
SUPPORTED_ALGORITHMS = ("ES256", "RS256", "EdDSA")

_A2A_EXTRAS_HINT = (
    "A2A Agent Card verification requires the optional 'a2a' extras: "
    'pip install "cdzzy-agentlink[a2a]" (pyjwt + cryptography)'
)


class A2AVerificationError(RuntimeError):
    """Raised when an Agent Card signature fails verification."""


# ── Encoding helpers ─────────────────────────────────────────────────────────


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _b64url_uint(value: int) -> str:
    """Big-endian Base64url of an unsigned integer, minimal octets (RFC 7518)."""
    size = max(1, (value.bit_length() + 7) // 8)
    return _b64url_encode(value.to_bytes(size, "big"))


# ── Canonicalization ─────────────────────────────────────────────────────────


def canonical_card_bytes(card_data: Union[Dict[str, Any], AgentCard]) -> bytes:
    """
    Deterministic JSON canonicalization of the card with ``signatures`` omitted.

    This is the signed payload for both :func:`sign_agent_card` and
    :func:`verify_agent_card`. See the module docstring for the JCS caveat.
    """
    if isinstance(card_data, AgentCard):
        card_data = card_data.to_dict()
    data = {k: v for k, v in card_data.items() if k != "signatures"}
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


# ── Key management ───────────────────────────────────────────────────────────


def _require_deps() -> None:
    try:
        import cryptography  # noqa: F401
        import jwt  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(_A2A_EXTRAS_HINT) from e


def _load_private_key(private_pem: bytes):
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    return load_pem_private_key(private_pem, password=None)


def _public_key_to_jwk(public_key: Any, kid: str, algorithm: str) -> Dict[str, Any]:
    """Serialize a cryptography public key into a single JWK dict."""
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    jwk: Dict[str, Any] = {
        "kid": kid,
        "alg": algorithm,
        "use": "sig",
        "key_ops": ["verify"],
    }
    if isinstance(public_key, rsa.RSAPublicKey):
        numbers = public_key.public_numbers()
        jwk.update({"kty": "RSA", "n": _b64url_uint(numbers.n), "e": _b64url_uint(numbers.e)})
    elif isinstance(public_key, ec.EllipticCurvePublicKey):
        numbers = public_key.public_numbers()
        crv = "P-256" if isinstance(public_key.curve, ec.SECP256R1) else public_key.curve.name
        jwk.update({
            "kty": "EC",
            "crv": crv,
            "x": _b64url_uint(numbers.x),
            "y": _b64url_uint(numbers.y),
        })
    elif isinstance(public_key, ed25519.Ed25519PublicKey):
        raw = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        jwk.update({"kty": "OKP", "crv": "Ed25519", "x": _b64url_encode(raw)})
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported public key type: {type(public_key).__name__}")
    return jwk


def private_key_to_jwks(private_pem: bytes, kid: str, algorithm: str) -> Dict[str, Any]:
    """Derive the JWKS verification document from a private PEM key."""
    _require_deps()
    private_key = _load_private_key(private_pem)
    return {"keys": [_public_key_to_jwk(private_key.public_key(), kid, algorithm)]}


def generate_signing_key(
    algorithm: str = "ES256",
    kid: Optional[str] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Generate a fresh signing key pair for self-signed development cards.

    Returns:
        ``(private_pem, jwks)`` — PEM-encoded private key and the matching
        JWKS verification document.

    Raises:
        ValueError: If ``algorithm`` is not one of :data:`SUPPORTED_ALGORITHMS`.
    """
    _require_deps()
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm {algorithm!r}; expected one of {SUPPORTED_ALGORITHMS}"
        )
    kid = kid or f"a2a-{uuid.uuid4().hex[:12]}"

    from cryptography.hazmat.primitives import serialization

    if algorithm == "ES256":
        from cryptography.hazmat.primitives.asymmetric import ec

        key = ec.generate_private_key(ec.SECP256R1())
    elif algorithm == "RS256":
        from cryptography.hazmat.primitives.asymmetric import rsa

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    else:  # EdDSA
        from cryptography.hazmat.primitives.asymmetric import ed25519

        key = ed25519.Ed25519PrivateKey.generate()

    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return private_pem, private_key_to_jwks(private_pem, kid, algorithm)


# ── ES256 raw (JWS) <-> DER (cryptography) conversion ────────────────────────


def _der_to_raw(der_signature: bytes, size: int = 32) -> bytes:
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

    r, s = decode_dss_signature(der_signature)
    return r.to_bytes(size, "big") + s.to_bytes(size, "big")


def _raw_to_der(raw_signature: bytes) -> bytes:
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

    half = len(raw_signature) // 2
    r = int.from_bytes(raw_signature[:half], "big")
    s = int.from_bytes(raw_signature[half:], "big")
    return encode_dss_signature(r, s)


# ── Signing ──────────────────────────────────────────────────────────────────


def _sign_bytes(data: bytes, private_pem: bytes, algorithm: str) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa

    key = _load_private_key(private_pem)
    if algorithm == "ES256":
        if not isinstance(key, ec.EllipticCurvePrivateKey):  # pragma: no cover
            raise ValueError("ES256 requires an EC private key")
        der = key.sign(data, ec.ECDSA(hashes.SHA256()))
        return _der_to_raw(der)
    if algorithm == "RS256":
        if not isinstance(key, rsa.RSAPrivateKey):  # pragma: no cover
            raise ValueError("RS256 requires an RSA private key")
        return key.sign(data, padding.PKCS1v15(), hashes.SHA256())
    # EdDSA
    return key.sign(data)


def sign_agent_card(
    card: Union[AgentCard, Dict[str, Any]],
    private_pem: bytes,
    kid: Optional[str] = None,
    algorithm: str = "ES256",
) -> Union[AgentCard, Dict[str, Any]]:
    """
    Append a detached-JWS signature over the card (``signatures[]``).

    The card object is modified in place when an :class:`AgentCard` is given
    (and returned); a plain dict input returns a new dict carrying
    ``signatures``. The signed payload is :func:`canonical_card_bytes`, i.e.
    the card without its ``signatures`` field.
    """
    _require_deps()
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm {algorithm!r}; expected one of {SUPPORTED_ALGORITHMS}"
        )

    header = {"alg": algorithm, "typ": "JOSE"}
    if kid:
        header["kid"] = kid
    protected = _b64url_encode(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    signing_input = (protected + "." + _b64url_encode(canonical_card_bytes(card))).encode("ascii")
    signature = _b64url_encode(_sign_bytes(signing_input, private_pem, algorithm))

    entry = AgentCardSignature(protected=protected, signature=signature)
    if isinstance(card, AgentCard):
        card.signatures.append(entry)
        return card
    out = dict(card)
    out["signatures"] = list(out.get("signatures", [])) + [entry.to_dict()]
    return out


# ── Verification ─────────────────────────────────────────────────────────────


def _public_key_from_jwk(jwk: Dict[str, Any]):
    import jwt

    return jwt.PyJWK.from_dict(dict(jwk)).key


def _verify_bytes(data: bytes, signature: bytes, public_key: Any, algorithm: str) -> None:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, padding

    try:
        if algorithm == "ES256":
            public_key.verify(_raw_to_der(signature), data, ec.ECDSA(hashes.SHA256()))
        elif algorithm == "RS256":
            public_key.verify(signature, data, padding.PKCS1v15(), hashes.SHA256())
        else:  # EdDSA
            public_key.verify(signature, data)
    except InvalidSignature as e:
        raise A2AVerificationError("Agent Card signature does not match the JWKS key") from e


def verify_agent_card(
    card: Union[AgentCard, Dict[str, Any]],
    jwks: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    kid: Optional[str] = None,
    require_signature: bool = True,
) -> bool:
    """
    Verify every ``signatures[]`` entry of an inbound Agent Card against a JWKS.

    Args:
        card: :class:`AgentCard` instance or card dict (as received).
        jwks: JWKS document ``{"keys": [...]}`` or a single JWK dict.
        kid: Optionally restrict verification to this key id.
        require_signature: When True (default), a card without any
            ``signatures`` entry is rejected.

    Returns:
        True when all signatures verify.

    Raises:
        A2AVerificationError: On a missing signature, an unsupported algorithm,
            an unknown ``kid``, or any signature that does not verify.
    """
    _require_deps()
    data = card.to_dict() if isinstance(card, AgentCard) else dict(card)
    signatures: List[Dict[str, Any]] = [
        s for s in (data.get("signatures") or []) if isinstance(s, dict)
    ]
    if not signatures:
        if require_signature:
            raise A2AVerificationError("Agent Card carries no signatures")
        return True

    if isinstance(jwks, dict):
        keys: List[Dict[str, Any]] = list(jwks.get("keys") or [])
        if not keys and jwks.get("kty"):
            keys = [jwks]
    else:
        keys = [k for k in jwks if isinstance(k, dict)]
    if not keys:
        raise A2AVerificationError("JWKS carries no verification keys")

    payload = canonical_card_bytes(data)
    for sig in signatures:
        protected = str(sig.get("protected", ""))
        signature_b64 = str(sig.get("signature", ""))
        try:
            header = json.loads(_b64url_decode(protected))
            algorithm = header.get("alg")
            sig_kid = header.get("kid")
        except (ValueError, json.JSONDecodeError) as e:
            raise A2AVerificationError(f"Malformed JWS protected header: {e}") from e
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise A2AVerificationError(f"Unsupported JWS algorithm: {algorithm!r}")
        if kid is not None and sig_kid is not None and sig_kid != kid:
            raise A2AVerificationError(
                f"Signature kid {sig_kid!r} does not match the expected kid {kid!r}"
            )

        # RFC 7515: ``kid`` in the JWS header is only a hint, never a hard
        # requirement.  A signature header without ``kid`` must be tried
        # against every key of the set (the correct one is found by probing).
        if kid is not None:
            # Caller restricted verification to one key id (or key-less entries).
            candidates = [k for k in keys if k.get("kid") in (None, kid)]
        elif sig_kid is None:
            candidates = list(keys)
        else:
            candidates = [k for k in keys if k.get("kid") in (None, sig_kid)]
        if not candidates:
            raise A2AVerificationError(f"No JWKS key matches kid {sig_kid!r}")

        signing_input = (protected + "." + _b64url_encode(payload)).encode("ascii")
        raw_signature = _b64url_decode(signature_b64)
        last_error: Optional[Exception] = None
        for jwk in candidates:
            try:
                public_key = _public_key_from_jwk(jwk)
                _verify_bytes(signing_input, raw_signature, public_key, algorithm)
                break
            except A2AVerificationError as e:
                last_error = e
        else:
            raise A2AVerificationError(
                f"Agent Card signature verification failed (kid={sig_kid!r}, alg={algorithm})"
            ) from last_error
    return True
