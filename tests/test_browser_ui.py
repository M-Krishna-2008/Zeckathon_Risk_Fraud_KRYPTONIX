"""
Browser UI End-to-End Test and Screenshot Suite.

Launches the application server, initializes headless Chrome via Selenium,
exercises all UI functionality (suggestion cards, dark/light theme, new chat,
chat history, and responsive mobile layout), and captures screenshots into reports/screenshots/.
"""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    webdriver = None

BASE_DIR = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = BASE_DIR / "reports" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_server(url: str = "http://127.0.0.1:8000", timeout: int = 30) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def run_browser_ui_tests():
    print("===================================================================")
    print("Starting Browser UI End-to-End Test Suite")
    print("===================================================================")

    if webdriver is None:
        print("[Notice] Selenium is not installed in this environment. Skipping browser UI tests.")
        return

    # 1. Start application server
    print("[1/6] Launching Distress Watch backend server...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.server:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not wait_for_server():
            raise RuntimeError("Backend server failed to start within 30 seconds.")
        print("  -> Server verified responsive at http://127.0.0.1:8000")

        # 2. Setup Selenium Chrome WebDriver
        print("[2/6] Initializing Headless Chrome WebDriver...")
        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1280,950")
        opts.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=opts)

        wait = WebDriverWait(driver, 20)

        try:
            # 3. Test Home View (Dark Theme)
            print("[3/6] Testing Home View and Theme Toggle...")
            driver.get("http://127.0.0.1:8000")
            wait.until(EC.presence_of_element_located((By.ID, "emptyState")))
            time.sleep(1)

            p1 = SCREENSHOTS_DIR / "01_home_dark.png"
            driver.save_screenshot(str(p1))
            print(f"  -> Saved screenshot: {p1.name}")

            # Toggle to Light Theme
            theme_btn = driver.find_element(By.ID, "themeToggleBtn")
            theme_btn.click()
            time.sleep(0.5)
            assert driver.find_element(By.TAG_NAME, "html").get_attribute("data-theme") == "light", "Failed to switch to light theme"
            p2 = SCREENSHOTS_DIR / "02_home_light.png"
            driver.save_screenshot(str(p2))
            print(f"  -> Saved screenshot: {p2.name}")

            # Toggle back to Dark Theme
            theme_btn.click()
            time.sleep(0.5)
            assert driver.find_element(By.TAG_NAME, "html").get_attribute("data-theme") == "dark", "Failed to switch back to dark theme"

            # 4. Test Suggestion Cards & Query Answers
            print("[4/6] Testing Suggestion Cards and Query Routing...")
            suggestions = [
                ("Show Watch-List", "show watch-list", "03_suggestion_watchlist.png", "Top 40"),
                ("Why was Jet Airways flagged?", "why was Jet Airways flagged?", "04_suggestion_jet_airways.png", "Jet Airways"),
                ("Recall & Performance", "recall and model performance", "05_suggestion_recall_metrics.png", "Recall"),
                ("Policy & Cost Model", "credit policy and cost threshold", "06_suggestion_policy_cost.png", "0.0476"),
            ]

            for idx, (title, prompt, shot_name, expected_text) in enumerate(suggestions, 1):
                print(f"  -> Testing Suggestion {idx}: '{title}' ...")
                # If not first, click 'New Chat'
                if idx > 1:
                    new_chat_btn = wait.until(EC.element_to_be_clickable((By.ID, "newChatBtn")))
                    new_chat_btn.click()
                    time.sleep(0.5)

                # Locate card by data-prompt
                card = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, f".suggestion-card[data-prompt='{prompt}']")))
                card.click()

                # Wait for assistant row to appear
                wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, ".message-row.assistant-row .assistant-body")) > 0)
                # Wait for typewriter effect / search to finish
                wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, ".typing-cursor")) == 0)
                time.sleep(1.5)

                # Verify response text contains expected keyword
                assistant_text = driver.find_element(By.CSS_SELECTOR, ".message-row.assistant-row .assistant-body").text
                assert expected_text in assistant_text, f"Response did not contain '{expected_text}', got: {assistant_text[:100]}..."

                shot_path = SCREENSHOTS_DIR / shot_name
                driver.save_screenshot(str(shot_path))
                print(f"     [OK] Captured '{shot_name}' (verified keyword '{expected_text}')")

            # 5. Test Chat History Navigation
            print("[5/6] Testing Chat History Sidebar Persistence...")
            history_items = driver.find_elements(By.CSS_SELECTOR, ".history-item")
            print(f"  -> Found {len(history_items)} chat sessions in history.")
            assert len(history_items) >= 4, f"Expected at least 4 chat sessions in history list, found {len(history_items)}"

            # Click second chat in history to verify navigation
            history_items[1].click()
            time.sleep(0.8)
            p7 = SCREENSHOTS_DIR / "07_chat_history_switched.png"
            driver.save_screenshot(str(p7))
            print(f"  -> Saved screenshot: {p7.name}")

            # 6. Test Mobile Viewport and Drawer
            print("[6/6] Testing Mobile Viewport and Drawer Navigation...")
            driver.set_window_size(390, 844)  # Mobile viewport (iPhone / Android)
            time.sleep(0.5)

            # Open mobile menu
            menu_btn = wait.until(EC.element_to_be_clickable((By.ID, "mobileMenuBtn")))
            menu_btn.click()
            time.sleep(0.5)

            p8 = SCREENSHOTS_DIR / "08_mobile_drawer.png"
            driver.save_screenshot(str(p8))
            print(f"  -> Saved screenshot: {p8.name}")

            # Close drawer
            close_btn = wait.until(EC.element_to_be_clickable((By.ID, "sidebarCloseBtn")))
            close_btn.click()
            time.sleep(0.5)

            p9 = SCREENSHOTS_DIR / "09_mobile_chat_view.png"
            driver.save_screenshot(str(p9))
            print(f"  -> Saved screenshot: {p9.name}")

            print("\nAll browser UI tests PASSED successfully!")
            print(f"Captured {len(list(SCREENSHOTS_DIR.glob('*.png')))} screenshots in {SCREENSHOTS_DIR}")

        finally:
            driver.quit()

    finally:
        server_process.terminate()
        server_process.wait()
        print("Backend server shutdown complete.")


if __name__ == "__main__":
    run_browser_ui_tests()
