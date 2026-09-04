import pandas as pd
from playwright.sync_api import sync_playwright

CATEGORY_KEYWORDS = {
    "Health": ["fda", "drug", "vaccine", "hospital", "medical", "disease", "patient", "biotech", "cancer", "clinical trial", "health", "pharma", "therapeutics", "biopharma", "surgical", "diagnostics", "pfizer", "moderna", "johnson & johnson", "merck", "abbvie", "amgen", "eli lilly", "bristol myers", "novartis", "roche", "sanofi", "boston scientific", "stryker", "medtronic", "intuitive surgical", "biolife", "dentsply", "neuropace", "gsk", "astrazeneca", "regeneron", "vertex pharmaceuticals"],
    "Energy": ["oil", " gas", "solar", "renewable", "nuclear", "power grid", "lng", "drilling", "energy", "opec", "barrel"],
    "Technology": ["ai ", " ai,", "artificial intelligence", "chip", "semiconductor", "software", " app", "cyber", "robot", "quantum", "tech", "iphone", "cloud", "data center", "ipad", "macbook"],
    "Politics": ["senate", "congress", "election", "president", "minister", "tariff", "sanctions", " war", "government", "regulation", "ceasefire", "gaza", "ukraine", "fcc"],
    "Markets": ["stock", "shares", "nasdaq", "dow jones", "s&p", "ipo", "etf", "dividend", "futures", "yields", "bond", "gold", "currency", "forex", "crypto", "bitcoin", "index", "rally", "selloff", "rate cut", "fed "],
}


def classify_category(title):
    lowered = title.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "Business"


def scrape_fool(page):
    page.goto("https://www.fool.com/investing/", wait_until="domcontentloaded")
    articles = page.locator("a:has(h5)").all()

    seen_urls = set()
    results = []
    for a in articles:
        href = a.get_attribute("href")
        url = "https://www.fool.com" + href
        if url in seen_urls:
            continue
        seen_urls.add(url)
        title = a.locator("h5").inner_text()
        results.append({"url": url, "title": title})
    return results


def scrape_marketscreener(context):
    page = context.new_page()
    page.goto("https://www.marketscreener.com/news/", wait_until="domcontentloaded")
    articles = page.locator("a:has(b)").all()

    seen_urls = set()
    results = []
    for a in articles:
        href = a.get_attribute("href")
        url = "https://www.marketscreener.com" + href
        if url in seen_urls:
            continue
        seen_urls.add(url)
        title = a.locator("b").inner_text()
        results.append({"url": url, "title": title})
    page.close()
    return results


def scrape_tradingview(page):
    page.goto("https://www.tradingview.com/news/", wait_until="domcontentloaded")
    articles = page.locator("a:has([data-qa-id='news-headline-title'])").all()

    seen_urls = set()
    results = []
    for a in articles:
        href = a.get_attribute("href")
        url = "https://www.tradingview.com" + href
        if url in seen_urls:
            continue
        seen_urls.add(url)
        title = a.locator("[data-qa-id='news-headline-title']").inner_text()
        results.append({"url": url, "title": title})
    return results


def scrape_yahoo_finance_healthcare(page):
    page.goto("https://finance.yahoo.com/sectors/healthcare/", wait_until="domcontentloaded")
    articles = page.locator("a:has(h3)").all()

    seen_urls = set()
    results = []
    for a in articles:
        href = a.get_attribute("href")
        if not href:
            continue
        if href in seen_urls:
            continue
        seen_urls.add(href)
        title = a.locator("h3").inner_text()
        results.append({"url": href, "title": title})
    return results


all_rows = []

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()

    print("Scraping fool.com...")
    all_rows.extend(scrape_fool(page))

    print("Scraping tradingview.com...")
    all_rows.extend(scrape_tradingview(page))

    print("Scraping finance.yahoo.com healthcare sector...")
    all_rows.extend(scrape_yahoo_finance_healthcare(page))

    browser.close()

    # marketscreener.com needs a real user-agent to bypass bot detection
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        viewport={"width": 1280, "height": 800},
    )
    print("Scraping marketscreener.com...")
    all_rows.extend(scrape_marketscreener(context))

    browser.close()

print(f"Scraped {len(all_rows)} rows this run")

new_df = pd.DataFrame(all_rows).rename(columns={"url": "URL", "title": "Title"})

existing = pd.read_csv("../data/news_dataset.csv")
combined = pd.concat([existing[["URL", "Title"]], new_df], ignore_index=True)
combined = combined.drop_duplicates(subset="URL").reset_index(drop=True)

combined["Category"] = combined["Title"].apply(classify_category)

combined.to_csv("../data/news_dataset.csv", index=False)
print(f"Saved {len(combined)} total rows to ../data/news_dataset.csv")
print()
print("Category breakdown:")
print(combined["Category"].value_counts())
