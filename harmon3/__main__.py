"""Entry point.

With no arguments this launches the GUI. The verification flags run headless, without a
QApplication, so the graph contract can be checked from a terminal or a CI step.
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harmon3",
        description="PySide6 front-end for the MiniMax H3 reference-to-video ComfyUI workflow.",
    )
    parser.add_argument("--server", default=None,
                        help=f"ComfyUI base URL (default: from settings, else {config.DEFAULT_SERVER_URL})")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    modes = parser.add_argument_group("verification (headless)")
    modes.add_argument("--dry-run", action="store_true",
                       help="print the graph that would be submitted")
    modes.add_argument("--diff", action="store_true",
                       help="with --dry-run, diff it against the shipped workflow instead")
    modes.add_argument("--check", action="store_true",
                       help="validate the graph against the server's node schemas")
    modes.add_argument("--roles", action="store_true",
                       help="show which node in the workflow plays which role")
    modes.add_argument("--upload-test", metavar="FILE",
                       help="upload a file and confirm it is retrievable")
    modes.add_argument("--object-info", action="store_true",
                       help="dump the node schemas this app depends on")

    posing = parser.add_argument_group("pose estimation (headless)")
    posing.add_argument("--pose", metavar="VIDEO",
                        help="render a clip's skeleton to --out, the way the Pose toggle "
                             "would; the fastest way to judge the estimator without the GUI")
    posing.add_argument("--out", metavar="FILE", help="where --pose writes its clip")
    posing.add_argument("--start", type=int, default=0,
                        help="first frame of the section to pose (default: 0)")
    posing.add_argument("--frames", type=int, default=None,
                        help="how many frames to pose (default: the saved duration)")
    return parser


def resolve_server(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        from .settings import load_settings
        return load_settings().get("server_url") or config.DEFAULT_SERVER_URL
    except Exception:
        return config.DEFAULT_SERVER_URL


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        config.load_workflow()
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 2

    server = resolve_server(args.server)

    if args.pose:
        from . import cli
        return cli.cmd_pose(args.pose, args.out, args.start, args.frames)

    if args.dry_run or args.check or args.roles or args.upload_test or args.object_info:
        from . import cli
        if args.dry_run:
            return cli.cmd_dry_run(args.diff)
        if args.check:
            return cli.cmd_check(server)
        if args.roles:
            return cli.cmd_roles()
        if args.upload_test:
            return cli.cmd_upload_test(server, args.upload_test)
        return cli.cmd_object_info_dump(server)

    from .ui.app import run_gui
    return run_gui(server)


if __name__ == "__main__":
    sys.exit(main())
