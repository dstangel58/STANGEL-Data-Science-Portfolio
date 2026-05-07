# 📚 **Comic Book Valuation: Tidy Data & Market Trends** 📚

## **Project Summary:**
In this project, you can explore the relationships between the average value of comic book appearances over four decades across different market platforms. By applying Tidy Data principles, the raw dataset was transformed from a "wide" to a "narrow" format using pandas.melt(), ensuring each row represents a single observation. The analysis compares four major valuation methods—Heritage Auctions, eBay, Wizard (Wiz), and O Street—to visualize how market prices fluctuate over time.

While Heritage Auctions tracks the highest individual sale prices, the other platforms aggregate average sale values. The data was cleaned by stripping special characters ($, ,) and splitting columns to follow structured data standards.

## 📖 **Steps to Use:** 📖
* Clone the Repository
    * git clone https://github.com/dstangel58/STANGEL-Data-Science-Portfolio.git
    * cd STANGEL-Data-Science-Portfolio/ComicValuationApp
* Download requirements.txt
    * (Requires: pandas, seaborn, matplotlib)
    * pip install -r requirements.txt
* Run the code!
    * python comic_analysis.py

## ⬇️ **Valuation Sources:** ⬇️
* Heritage Auctions: Highest individual sale metrics.
* eBay: Aggregated average market sales.
* Wizard (Wiz): Historical guide pricing.
* O Street: Secondary market average valuations.

## 🧑‍🏫 **Guides and Tutorials:** 🧑‍🏫
* Tidy Data Paper (Hadley Wickham): https://vita.had.co.nz/papers/tidy-data.pdf
* Pandas Melt Guide: https://pandas.pydata.org/docs/reference/api/pandas.melt.html
* Seaborn Barplot Documentation: https://seaborn.pydata.org/generated/seaborn.barplot.html
* Pivot Table Medians: https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html

📸 **Images:**##  📸
![alt text](<Screenshot 2026-05-06 at 11.24.47 PM.png>)
Cleaned Data