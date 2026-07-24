import unittest
from unittest.mock import patch

import llm_decode_bench as bench


SAFE_DWORDS = (
    "ForceP2P=0x11;"
    "GrdmaPciTopoCheckOverride=1;"
    "EnableResizableBar=1"
)
UNSAFE_DWORDS = (
    "ForceP2P=0x11;"
    "RMForceP2PType=1;"
    "RMPcieP2PType=2;"
    "GrdmaPciTopoCheckOverride=1;"
    "EnableResizableBar=1"
)


def detect(params_dwords: str, modprobe_dwords: str):
    params = (
        f'RegistryDwords: "{params_dwords}"\n'
        "DmaRemapPeerMmio: 1\n"
    )
    modprobe = (
        'options nvidia NVreg_RegistryDwords="'
        f'{modprobe_dwords}"\n'
    )
    with patch.object(
        bench,
        "_read_text_file",
        side_effect=lambda path: params if path.startswith("/proc/") else modprobe,
    ):
        return bench.detect_nvidia_p2p_override()


class NvidiaP2POverrideTests(unittest.TestCase):
    def test_safe_three_key_override_is_registry_ready(self):
        status = detect(SAFE_DWORDS, SAFE_DWORDS)

        self.assertTrue(status["registry_ready"])
        self.assertTrue(status["effective"])
        self.assertTrue(status["configured"])
        self.assertEqual(status["unsafe_runtime"], {})
        self.assertEqual(status["unsafe_configured"], [])
        self.assertFalse(status["data_path_verified"])
        self.assertIn("run P2PMark", bench.p2p_override_summary(status))

    def test_legacy_selector_pair_is_rejected(self):
        status = detect(UNSAFE_DWORDS, UNSAFE_DWORDS)

        self.assertFalse(status["registry_ready"])
        self.assertFalse(status["effective"])
        self.assertFalse(status["configured"])
        self.assertEqual(
            status["unsafe_runtime"],
            {"RMForceP2PType": "1", "RMPcieP2PType": "2"},
        )
        self.assertEqual(
            status["unsafe_configured"],
            ["RMForceP2PType", "RMPcieP2PType"],
        )
        self.assertNotIn("RMForceP2PType", status["suggested_modprobe_line"])
        self.assertNotIn("RMPcieP2PType", status["suggested_modprobe_line"])
        self.assertIn("unsafe legacy selectors", bench.p2p_override_summary(status))

    def test_missing_required_key_is_reported(self):
        incomplete = "ForceP2P=0x11;EnableResizableBar=1"
        status = detect(incomplete, incomplete)

        self.assertFalse(status["registry_ready"])
        self.assertEqual(status["missing"], ["GrdmaPciTopoCheckOverride"])

    def test_successful_p2pmark_verifies_data_path(self):
        status = detect(SAFE_DWORDS, SAFE_DWORDS)

        bench.record_p2pmark_verification(
            status,
            {"status": "ok", "mode": "all"},
        )

        self.assertTrue(status["data_path_verified"])
        self.assertEqual(status["data_path_status"], "verified")
        self.assertEqual(status["data_path_mode"], "all")

    def test_failed_p2pmark_does_not_verify_data_path(self):
        status = detect(SAFE_DWORDS, SAFE_DWORDS)

        bench.record_p2pmark_verification(
            status,
            {"status": "failed", "mode": "bandwidth"},
        )

        self.assertFalse(status["data_path_verified"])
        self.assertEqual(status["data_path_status"], "failed")
        self.assertEqual(status["data_path_mode"], "bandwidth")


if __name__ == "__main__":
    unittest.main()
