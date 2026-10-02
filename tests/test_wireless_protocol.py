"""Fake ESP32-S3 on localhost: HELLO/CHALLENGE/START handshake + BWIM UDP stream parsed by the real worker."""
import os
import socket
import struct
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

import robotic_arm as ra

KEY = "unit-test-key"


class FakeDevice(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", ra.WIFI_CONTROL_PORT))
        self.sock.settimeout(0.2)
        self.stop_evt = threading.Event()
        self.client = None
        self.seq = 0

    def run(self):
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while not self.stop_evt.is_set():
            try:
                data, addr = self.sock.recvfrom(1024)
            except socket.timeout:
                data = None
            if data:
                parts = data.decode().split("|")
                if parts[0] == "DISCOVER":
                    self.sock.sendto(b"HELLO|id1|FakeBand|STA|127.0.0.1|1|0|fw-test", addr)
                elif parts[0] == "CHALLENGE":
                    self.sock.sendto(b"CHALLENGE|abc123", addr)
                elif parts[0] == "START":
                    assert parts[-1] == ra.sign_message(KEY, "START", "abc123", *parts[2:-1])
                    self.client = (parts[2], int(parts[3]))
                    self.sock.sendto(b"ACK|STARTED", addr)
            if self.client:
                frames = b""
                for _ in range(ra.WIRELESS_FRAMES_PER_PACKET):
                    frames += struct.pack(ra.WIRELESS_FRAME_FORMAT, self.seq, self.seq * 2000, self.seq, self.seq * 2000,
                                          *range(100, 108), 10.0, 20.0, 30.0, 1)
                header = struct.pack(ra.WIFI_PACKET_HEADER_FORMAT, b"BWIM", 1, ra.WIRELESS_FRAMES_PER_PACKET,
                                     ra.WIRELESS_FRAME_SIZE, self.seq)
                tx.sendto(header + frames, self.client)
                self.seq += 1
                time.sleep(0.01)


@pytest.mark.skipif(not ra.HAS_QT, reason="PyQt5 not installed")
def test_authenticated_start_and_stream_parse():
    dev = FakeDevice()
    dev.start()
    app = ra.QApplication.instance() or ra.QApplication([])
    got = []
    worker = ra.WirelessStreamWorker(ra.WIFI_STREAM_PORT)
    worker.batch_received.connect(got.append)
    worker.start()
    try:
        ack = ra.ControlProtocol.start_stream("127.0.0.1", KEY, "127.0.0.1", ra.WIFI_STREAM_PORT)
        assert ack == ["STARTED"]
        t_end = time.monotonic() + 3
        while len(got) < 5 and time.monotonic() < t_end:
            app.processEvents()
            time.sleep(0.02)
        assert len(got) >= 5
        b = got[0]
        assert b.samples.shape == (ra.WIRELESS_FRAMES_PER_PACKET, ra.WIRELESS_TOTAL_CHANNELS)
        assert list(b.samples[0, :8]) == [100, 101, 102, 103, 104, 105, 106, 107]
        assert list(b.samples[0, 8:]) == [10.0, 20.0, 30.0]
        assert worker.stats.packet_loss_percent < 5
    finally:
        worker.stop()
        dev.stop_evt.set()
        dev.join(timeout=2)
        dev.sock.close()
