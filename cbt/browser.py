"""Headless Chromium and local HTML URLs shared by the exporters."""
from contextlib import contextmanager
from pathlib import Path


class BrowserUnavailable(RuntimeError):
    """Optional Playwright or its browser is unavailable."""


def file_url(path, capture=False):
    return Path(path).resolve().as_uri() + ("?capture" if capture else "")


@contextmanager
def chromium_browser(args=()):
    hint = "install Playwright and run python -m playwright install chromium"
    try:
        from playwright.sync_api import Error, sync_playwright
    except ImportError as exc:
        raise BrowserUnavailable(hint) from exc
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(args=list(args))
        except Error as exc:
            raise BrowserUnavailable(f"{hint} ({str(exc).splitlines()[0]})") from exc
        try:
            yield browser
        finally:
            browser.close()
