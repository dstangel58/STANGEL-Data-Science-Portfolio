### 🧬 PROJECT OVERVIEW ### 
In this project, I transform the raw dataset to track the relationships between the average value of comic book appearances over four decades and two categorical variables: platform and decade. By using tidy data principles, I changed the "wide" format of the data to a "narrow" one by using the pandas.melt() function. The analysis explores how the values of certain character appearnces change over time, compared to four main valuation methods: 
1) Heritage Auctions
2) eBay
3) Wizard (Wiz)
4) O Street

While Heritage used highest individual sale price, the other platforms aggregated various average sale values over time. 

### Data Cleaning and Preparation ### 

The data was split into the 'Source' and 'Decade' columns to better organize results and follow the tidy data principle of one observation per row. In addition, I removed dollar signs and commas to standarize values.

### 📈 Insights ### 

The charts show higher values over time. Heritage is separated because it employs a different metric. I created bar charts by valuation and decade to show change over time on a standard scale. 

### Statistics and Pivot Tables ### 

Pivot tables were created using medians since they show a more typical value. 

MAIN CONCLUSION --> Older appearances have higher values

### Libraries Used ### 

Python, pandas, seaborn, matplotlib