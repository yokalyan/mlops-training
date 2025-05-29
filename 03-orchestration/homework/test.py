import argparse
from datetime import datetime
import sys

def get_current_year_month():
    today = datetime.today()
    return today.year, today.month

parser = argparse.ArgumentParser()

parser.add_argument("--year", type=int, default=None, help="Year (e.g. 2025)")
parser.add_argument("--month", type=int, default=None, help="Month (1-12)")

args = parser.parse_args()

# Validate partial input
if (args.year is None) != (args.month is None):
    print("Error: You must provide both --year and --month together, or neither.", file=sys.stderr)
    sys.exit(1)

# Assign defaults if neither was provided
if args.year is None and args.month is None:
    args.year, args.month = get_current_year_month()

print(f"Running for year: {args.year}, month: {args.month}")
