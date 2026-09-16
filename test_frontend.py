import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        console_logs = []
        page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))

        print("Navigating to frontend...")
        await page.goto("http://localhost:8000")
        await page.wait_for_timeout(2000)

        # 1. Health pill
        pill = await page.locator("#systemStatusText").text_content()
        print(f"1. Health Pill Status: {pill}")

        # 2. Render imagery
        await page.click("#btnStartProcessing")
        await page.wait_for_timeout(2000)
        scene_rendered = await page.evaluate("() => { return !!inputImageOverlay && !!srImageOverlay; }")
        print(f"2. Scene Rendered: {scene_rendered}")

        # 3. Upload valid tif
        await page.evaluate("() => { document.getElementById('navUpload').click(); }")
        await page.wait_for_timeout(1000)
        # Playwright file chooser handling
        async with page.expect_file_chooser() as fc_info:
            await page.click("#btnUploadGeoTIFF")
        file_chooser = await fc_info.value
        await file_chooser.set_files("P:\\TerraResolve Final\\terraresolve\\dummy.tif")

        # wait for job completion
        print("Waiting for job to complete...")
        for _ in range(15):
            await page.wait_for_timeout(2000)
            banner = await page.evaluate("() => document.getElementById('jobStatusBanner') ? document.getElementById('jobStatusBanner').innerText : ''")
            if "COMPLETED" in banner or "FAILED" in banner:
                print(f"3. Valid Upload Final Banner: {banner}")
                break

        # 4. Upload invalid tif (no_crs.tif)
        async with page.expect_file_chooser() as fc_info:
            await page.click("#btnUploadGeoTIFF")
        file_chooser = await fc_info.value
        await file_chooser.set_files("P:\\TerraResolve Final\\terraresolve\\no_crs.tif")

        print("Waiting for invalid job response...")
        for _ in range(5):
            await page.wait_for_timeout(1000)
            banner = await page.evaluate("() => document.getElementById('jobStatusBanner') ? document.getElementById('jobStatusBanner').innerText : ''")
            if "FAILED" in banner:
                print(f"4. Invalid Upload Final Banner: {banner}")
                break

        print("\nConsole Logs:")
        for log in console_logs:
            print(log)

        await browser.close()

asyncio.run(run())
