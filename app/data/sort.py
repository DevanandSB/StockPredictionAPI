import json
import yfinance as yf
from tqdm import tqdm  # Used for the progress bar


def get_market_cap(symbol: str) -> int:
    """
    Fetches the market capitalization for a given Indian stock symbol.
    It tries the National Stock Exchange (.NS) first, then the Bombay Stock Exchange (.BO).
    Returns an integer value of the market cap or 0 if not found.
    """
    try:
        # Create a Ticker object for the stock symbol. Append .NS for NSE.
        stock = yf.Ticker(f"{symbol}.NS")

        # 'info' is a dictionary containing all the stock's data
        market_cap = stock.info.get('marketCap')

        # If .NS fails, some stocks might only be on BSE (.BO)
        if not market_cap:
            stock_bse = yf.Ticker(f"{symbol}.BO")
            market_cap = stock_bse.info.get('marketCap')

        # Return the market cap if found, otherwise return 0
        return int(market_cap) if market_cap else 0

    except Exception:
        # If the symbol is invalid or another error occurs, return 0
        return 0


def main():
    """
    The main function to execute the script's logic.
    """
    # 1. Load the company data from the JSON file
    try:
        with open('companies.json', 'r', encoding='utf-8') as f:
            companies = json.load(f)
    except FileNotFoundError:
        print("Error: 'companies.json' not found. Please ensure the file is in the same directory.")
        return  # Exit the function if the file is not found

    # 2. Fetch market cap for each company and add it to the list
    print(f"Fetching market capitalization data for {len(companies)} companies. This may take a while...")

    # tqdm creates a smart progress bar to visualize the process
    for company in tqdm(companies, desc="Processing Companies"):
        market_cap_value = get_market_cap(company['symbol'])
        company['market_cap'] = market_cap_value

    # 3. Sort the list of companies by 'market_cap' in descending order
    # We use a lambda function as the key, which tells the sort function which value to use for comparison.
    # We also filter out any companies where the market cap could not be found (is 0).
    sorted_companies = sorted(
        [c for c in companies if c.get('market_cap', 0) > 0],
        key=lambda x: x['market_cap'],
        reverse=True
    )

    # 4. Save the newly sorted list to a new JSON file
    output_filename = 'companies_sorted_by_market_cap.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(sorted_companies, f, indent=4)

    print(f"\n✅ Sorting complete! The sorted data has been saved to '{output_filename}'")

    # 5. Display the top 10 companies as a confirmation
    print("\n--- Top 10 Companies by Market Capitalization ---")
    for i, company in enumerate(sorted_companies[:10]):
        # Formatting the market cap with commas for better readability
        mc_formatted = f"{company['market_cap']:,}"
        print(f"{i + 1:>2}. {company['name']:<45} ({company['symbol']}) - Market Cap: ₹{mc_formatted}")


# This block checks if the script is being run directly.
# If it is, it calls the main() function.
if __name__ == "__main__":
    main()