"""Terminal-key parsing and bounded decode-output preview contracts."""

import importlib.util
import io
import json
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "llm_decode_bench.py"
SPEC = importlib.util.spec_from_file_location("bench_viewer_tests", MODULE_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def render(viewer, width=90, height=10):
    output = io.StringIO()
    console = BENCH.Console(file=output, width=width, color_system=None)
    console.print(viewer.render(width, height))
    return output.getvalue()


class OutputViewerTests(unittest.TestCase):
    def test_hidden_by_default_and_can_be_toggled(self):
        viewer = BENCH.OutputViewer()
        self.assertFalse(viewer.visible)
        viewer.handle_key(" ")
        self.assertIsNone(viewer.snapshot)
        viewer.handle_key("o")
        self.assertTrue(viewer.visible)
        viewer.handle_key("o")
        self.assertFalse(viewer.visible)

    def test_pause_freezes_view_not_live_buffer(self):
        viewer = BENCH.OutputViewer()
        viewer.handle_key("o")
        viewer.feed(0, 0, "one", "reasoning", "Before pause.")
        viewer.handle_key(" ")
        self.assertIn("Before pause.", render(viewer))
        viewer.feed(0, 0, "one", "content", "Generated while paused.")
        self.assertNotIn("Generated while paused.", render(viewer))
        self.assertIn("Generated while paused.", "".join(viewer.parts))
        viewer.handle_key("end")
        self.assertIn("Generated while paused.", render(viewer))
        self.assertIn("LIVE", render(viewer))

    def test_paging_and_resume_are_independent_of_incoming_text(self):
        viewer = BENCH.OutputViewer()
        viewer.handle_key("o")
        viewer.feed(0, 0, "one", "content", "".join(f"line {i}\n" for i in range(200)))
        self.assertIn("line 199", render(viewer))
        viewer.handle_key("pageup")
        paused = render(viewer)
        self.assertNotIn("line 199", paused)
        viewer.feed(0, 0, "one", "content", "tail after paging\n")
        self.assertEqual(render(viewer), paused)
        viewer.handle_key("home")
        self.assertIn("line 0", render(viewer))
        viewer.handle_key("pagedown")
        self.assertNotIn("line 0\n", render(viewer))
        viewer.handle_key("end")
        self.assertIn("tail after paging", render(viewer))

    def test_selected_stream_and_channels_are_labeled_not_interleaved(self):
        viewer = BENCH.OutputViewer()
        viewer.start_cell(8, 16384)
        viewer.handle_key("o")
        viewer.feed(1, 0, "ignored", "content", "UNSELECTED_TEXT")
        viewer.feed(0, 0, "one", "reasoning", "Reasoning text.")
        viewer.feed(0, 0, "one", "content", "Answer text.")
        text = render(viewer, height=14)
        self.assertIn("stream 1/8", text)
        self.assertIn("[reasoning]", text)
        self.assertIn("[content]", text)
        self.assertNotIn("UNSELECTED_TEXT", text)
        viewer.handle_key("]")
        viewer.feed(1, 1, "two", "content", "Selected second worker.")
        text = render(viewer)
        self.assertIn("stream 2/8", text)
        self.assertNotIn("Answer text.", text)
        self.assertIn("Selected second worker.", text)

    def test_paused_snapshot_keeps_identity_across_cells(self):
        viewer = BENCH.OutputViewer()
        viewer.start_cell(8, 0)
        viewer.handle_key("o")
        viewer.feed(0, 0, "one", "content", "retained snapshot")
        viewer.handle_key(" ")
        viewer.start_cell(1, 32768)
        viewer.feed(0, 0, "two", "content", "following cell")
        self.assertIn("C=8", render(viewer))
        self.assertIn("retained snapshot", render(viewer))
        viewer.handle_key("end")
        self.assertIn("C=1", render(viewer))
        self.assertIn("following cell", render(viewer))

    def test_history_and_paused_snapshot_are_bounded(self):
        viewer = BENCH.OutputViewer()
        viewer.feed(0, 0, "one", "content", "x" * 100000)
        viewer.pause()
        for _ in range(100):
            viewer.feed(0, 0, "one", "content", "z" * 4096)
        self.assertEqual(viewer.chars, viewer.MAX_CHARS)
        self.assertEqual(len(viewer.snapshot), viewer.MAX_CHARS)
        self.assertGreater(viewer.dropped, 0)
        parts = len(viewer.parts)
        for _ in range(100):
            viewer.feed(0, 0, "one", "content", "\x00")
        self.assertEqual(len(viewer.parts), parts)

    def test_model_text_is_literal_and_cannot_control_terminal(self):
        viewer = BENCH.OutputViewer()
        viewer.feed(0, 0, "one", "content", "[bold red]untrusted[/bold red] \x1b[2J\x07 č漢🙂")
        output = render(viewer)
        self.assertIn("[bold red]untrusted[/bold red]", output)
        self.assertNotIn("\x1b", output)
        self.assertNotIn("\x07", output)
        self.assertIn("č漢🙂", output)

    def test_dashboard_panel_fits_terminal_and_loop_is_not_numeric(self):
        for width, height, mode in ((80, 24, "narrow"), (120, 40, "mid"), (180, 50, "wide")):
            state = BENCH.TUIState(
                total_tests=1, concurrency_levels=[1], context_lengths=[0],
                hw_monitor_enabled=False, metrics_available=False,
            )
            state.output_viewer.visible = True
            state.results[(0, 1)] = BENCH.LOOP_CELL_TPS
            state.output_viewer.feed(0, 0, "fixture", "content", "Hello from the model.\n")
            output = io.StringIO()
            console = BENCH.Console(file=output, width=width, height=height, color_system=None)
            with patch.object(BENCH, "live_layout_mode", return_value=(mode, width, height)):
                console.print(BENCH.build_display(state))
            text = output.getvalue()
            self.assertIn("Hello from the model.", text)
            self.assertIn("OUTPUT LIVE", text)
            self.assertIn("ERROR: loop", text)
            self.assertLessEqual(len(text.splitlines()), height)


class TerminalKeyTests(unittest.TestCase):
    def test_navigation_sequences_may_be_fragmented(self):
        parser = BENCH.TerminalKeyParser()
        self.assertEqual(parser.feed("o\x1b[5"), ["o"])
        self.assertEqual(parser.feed("~\x1b[6~\x1b["), ["pageup", "pagedown"])
        self.assertEqual(parser.feed("F \x1bOH"), ["end", " ", "home"])
        self.assertEqual(parser.feed("SQ[]"), ["s", "q", "[", "]"])

    def test_unknown_escape_sequences_do_not_trigger_benchmark_controls(self):
        parser = BENCH.TerminalKeyParser()
        self.assertEqual(parser.feed("\x1b[A\x1b[B\x1b[1;2S\x1bq"), [])
        self.assertEqual(parser.feed("o"), ["o"])

    def test_real_pty_keyboard_listener_and_terminal_restore(self):
        script = f"""
import importlib.util, json, sys, termios, time
spec = importlib.util.spec_from_file_location('pty_bench', {str(MODULE_PATH)!r})
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
fd = sys.stdin.fileno()
original = termios.tcgetattr(fd)
b.start_keyboard_listener(soft_quit=True)
deadline = time.monotonic() + 4
while termios.tcgetattr(fd)[3] & termios.ICANON:
    if time.monotonic() > deadline: raise RuntimeError('cbreak was not enabled')
    time.sleep(0.01)
print('ready', flush=True)
if not b._quit_event.wait(4): raise RuntimeError('quit key was not received')
keys = []
while not b._viewer_commands.empty(): keys.append(b._viewer_commands.get_nowait())
b._restore_terminal()
print(json.dumps({{'keys': keys, 'skipped': b._skip_event.is_set(),
                  'restored': termios.tcgetattr(fd) == original}}), flush=True)
"""
        master, slave = pty.openpty()
        process = None
        try:
            process = subprocess.Popen(
                [sys.executable, "-c", script], stdin=slave,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self.assertTrue(select.select([process.stdout], [], [], 5)[0])
            self.assertEqual(process.stdout.readline().strip(), "ready")
            os.write(master, b"o \x1b[5~\x1b[6~\x1b[F[]sq")
            output, error = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, error)
            result = json.loads(output)
            self.assertEqual(result["keys"], ["o", " ", "pageup", "pagedown", "end", "[", "]"])
            self.assertTrue(result["skipped"])
            self.assertTrue(result["restored"])
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.communicate()
            os.close(master)
            os.close(slave)


if __name__ == "__main__":
    unittest.main()
