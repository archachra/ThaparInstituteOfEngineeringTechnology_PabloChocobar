"""Airgap proof: make any attempt to open a network connection raise. Enabled with --airgap.
If prediction still works, nothing needed the network, so no video can have left the device."""
import socket


class AirgapViolation(RuntimeError):
    pass


_installed = {"n": 0}


def enable():
    def blocked(*a, **k):
        _installed["n"] += 1
        raise AirgapViolation("network access blocked: EchoGesture runs fully on-device")
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.getaddrinfo = blocked
    socket.create_connection = blocked


def blocked_attempts() -> int:
    return _installed["n"]
