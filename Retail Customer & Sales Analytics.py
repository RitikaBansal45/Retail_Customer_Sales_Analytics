#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# In[2]:


customers = pd.read_csv('C:/Users/Ritika/Downloads/Data set/Customer.csv')       
transactions = pd.read_csv('C:/Users/Ritika/Downloads/Data set/Transactions.csv')   
products=  pd.read_csv('C:/Users/Ritika/Downloads/Data set/prod_cat_info.csv') 


# In[3]:


print("  Customers   :", customers.shape)
print("  Products    :", products.shape)
print("  Transactions:", transactions.shape)


# In[4]:


transactions["tran_date"] = pd.to_datetime(transactions["tran_date"], dayfirst=True)
customers["DOB"]          = pd.to_datetime(customers["DOB"], dayfirst=True)


# In[5]:


ANALYSIS_DATE    = pd.Timestamp("2014-03-01")
customers["Age"] = ((ANALYSIS_DATE - customers["DOB"]).dt.days / 365).astype(int)


# In[6]:


transactions["is_return"] = transactions["Qty"] < 0


# In[7]:


df = transactions.merge(customers, left_on="cust_id", right_on="customer_Id", how="left")


# In[9]:


df = df.merge(products, left_on=["prod_cat_code", "prod_subcat_code"], right_on=["prod_cat_code", "prod_sub_cat_code"],
              how="left")


# In[10]:


sales   = df[df["is_return"] == False].copy()
returns = df[df["is_return"] == True].copy()


# In[11]:


print("\nTotal rows    :", len(df))
print("Sales rows    :", len(sales))
print("Return rows   :", len(returns))
print("Missing values:\n", df.isnull().sum()[df.isnull().sum() > 0])


# In[12]:


total_revenue = sales["total_amt"].sum()
total_orders  = sales["transaction_id"].nunique()
unique_buyers = sales["cust_id"].nunique()
avg_basket    = total_revenue / total_orders

print("\n── KEY METRICS ──────────────────────────")
print(f"  Gross Revenue   : Rs.{total_revenue:,.0f}")
print(f"  Total Orders    : {total_orders:,}")
print(f"  Unique Buyers   : {unique_buyers:,}")
print(f"  Avg Basket Size : Rs.{avg_basket:,.0f}")
print("─────────────────────────────────────────")


# In[13]:


cat_revenue = sales.groupby("prod_cat")["total_amt"].sum().sort_values(ascending=False)


# In[14]:


channel_revenue = sales.groupby("Store_type")["total_amt"].sum().sort_values(ascending=False)


# In[15]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cat_revenue.plot(kind="bar", ax=axes[0], color=sns.color_palette("Set2"))
axes[0].set_title("Revenue by Product Category")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Revenue (Rs.)")
axes[0].tick_params(axis="x", rotation=30)

channel_revenue.plot(kind="bar", ax=axes[1], color=sns.color_palette("Set2"))
axes[1].set_title("Revenue by Store Channel")
axes[1].set_xlabel("Channel")
axes[1].set_ylabel("Revenue (Rs.)")
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("01_revenue_category_channel.png")
plt.show()


# In[16]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Age distribution
axes[0].hist(customers["Age"].dropna(), bins=15, color="#5aacdb", edgecolor="white")
axes[0].set_title("Customer Age Distribution")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Count")

# Gender split
gender_counts = customers["Gender"].value_counts()
axes[1].pie(gender_counts, labels=["Male", "Female"], autopct="%1.1f%%",
            colors=["#66b3ff", "#ff9999"])
axes[1].set_title("Gender Split")

plt.tight_layout()
plt.savefig("02_demographics.png")
plt.show()


# In[18]:


monthly = sales.resample("M", on="tran_date")["total_amt"].sum()
rolling = monthly.rolling(3).mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(monthly.index, monthly / 1e6, width=20, color="#a8d8ea", alpha=0.7, label="Monthly Revenue")
ax.plot(monthly.index, rolling / 1e6, color="#e84393", linewidth=2, label="3-Month Rolling Avg")
ax.set_title("Monthly Revenue Trend (2011-2014)")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue (Rs. Millions)")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("03_monthly_trend.png")
plt.show()


# In[19]:


def calc_return_rate(group_col):
    total = df.groupby(group_col)["is_return"].count()
    ret   = df.groupby(group_col)["is_return"].sum()
    rate  = (ret / total * 100).round(2)
    return rate.sort_values(ascending=False)


# In[20]:


cat_return_rate     = calc_return_rate("prod_cat")
channel_return_rate = calc_return_rate("Store_type")


# In[21]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cat_return_rate.plot(kind="bar", ax=axes[0], color=sns.color_palette("Reds_d", len(cat_return_rate)))
axes[0].set_title("Return Rate by Category (%)")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Return Rate (%)")
axes[0].tick_params(axis="x", rotation=30)

channel_return_rate.plot(kind="bar", ax=axes[1], color=sns.color_palette("Reds_d", len(channel_return_rate)))
axes[1].set_title("Return Rate by Channel (%)")
axes[1].set_xlabel("Channel")
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("04_return_rates.png")
plt.show()


# In[22]:


rfm = sales.groupby("cust_id").agg(
    Recency   = ("tran_date",      lambda x: (ANALYSIS_DATE - x.max()).days),
    Frequency = ("transaction_id", "nunique"),
    Monetary  = ("total_amt",      "sum")
).reset_index()


# In[23]:


rfm["R_Score"] = pd.qcut(rfm["Recency"],  5, labels=[5, 4, 3, 2, 1])   # lower recency = better
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  5, labels=[1, 2, 3, 4, 5])


# In[24]:


def assign_segment(row):
    r = int(row["R_Score"])
    f = int(row["F_Score"])
    m = int(row["M_Score"])
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "New Customers"
    elif r <= 2 and f >= 3:
        return "At Risk"
    elif r <= 2 and f <= 2:
        return "Hibernating"
    else:
        return "Potential Loyalists"

rfm["Segment"] = rfm.apply(assign_segment, axis=1)

print("\nRFM Segment Summary:")
print(rfm.groupby("Segment")["cust_id"].count().sort_values(ascending=False))


# In[25]:


seg_counts = rfm["Segment"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
seg_counts.plot(kind="barh", ax=ax, color=sns.color_palette("Set2", len(seg_counts)))
ax.set_title("Customers per RFM Segment")
ax.set_xlabel("Number of Customers")
plt.tight_layout()
plt.savefig("05_rfm_segments.png")
plt.show()


# In[26]:


clv = sales.groupby("cust_id").agg(
    total_spend    = ("total_amt",      "sum"),
    orders         = ("transaction_id", "nunique"),
    first_purchase = ("tran_date",      "min"),
    last_purchase  = ("tran_date",      "max")
).reset_index()


# In[27]:


clv["lifespan_months"]  = ((clv["last_purchase"] - clv["first_purchase"]).dt.days / 30).clip(lower=1)
clv["avg_order_value"]  = clv["total_spend"] / clv["orders"]
clv["orders_per_month"] = clv["orders"] / clv["lifespan_months"]
clv["CLV"]              = clv["avg_order_value"] * clv["orders_per_month"] * clv["lifespan_months"]


# In[28]:


clv = clv.merge(rfm[["cust_id", "Segment"]], on="cust_id", how="left")


# In[29]:


clv_by_segment = clv.groupby("Segment")["CLV"].mean().sort_values(ascending=False)

print("\nAvg CLV by Segment:")
print(clv_by_segment.round(0))


# In[30]:


fig, ax = plt.subplots(figsize=(10, 5))
clv_by_segment.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(clv_by_segment)))
ax.set_title("Average CLV by RFM Segment")
ax.set_ylabel("Avg CLV (Rs.)")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("06_clv_by_segment.png")
plt.show()


# In[31]:


first_purchase      = sales.groupby("cust_id")["tran_date"].min().dt.to_period("M")
first_purchase.name = "cohort"


# In[ ]:





# In[32]:


cohort_df = sales.join(first_purchase, on="cust_id")
cohort_df["order_period"]  = cohort_df["tran_date"].dt.to_period("M")
cohort_df["period_offset"] = (cohort_df["order_period"] - cohort_df["cohort"]).apply(lambda x: x.n)


# In[33]:


cohort_matrix = (cohort_df.groupby(["cohort", "period_offset"])["cust_id"]
                           .nunique()
                           .unstack())


# In[34]:


retention = cohort_matrix.divide(cohort_matrix[0], axis=0).round(3) * 100
retention = retention.iloc[:, :13]


# In[35]:


fig, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(retention, annot=True, fmt=".0f", cmap="YlGnBu",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Retention (%)"})
ax.set_title("Monthly Cohort Retention (%)")
ax.set_xlabel("Months Since First Purchase")
ax.set_ylabel("Acquisition Cohort")
plt.tight_layout()
plt.savefig("07_cohort_retention.png")
plt.show()


# In[36]:


print("""
=========================================================
  KEY BUSINESS INSIGHTS & RECOMMENDATIONS
=========================================================

1. REVENUE MIX
   Books & Electronics drive ~47% of gross revenue.
   Action: Focus inventory and promotions here.

2. CHANNEL STRATEGY
   e-Shop leads in order volume.
   MBR has the highest average basket size.
   Action: Use e-Shop for acquisition, MBR for upselling.

3. CUSTOMER DEMOGRAPHICS
   Core buyers are aged 26-38. Gender split is near equal.
   Action: Target digital campaigns at the 26-35 bracket.

4. RETURN RATES
   ~9.4% of transactions are returns.
   Electronics & Clothing have the highest return rates.
   Action: Improve product descriptions and sizing guides.

5. RFM SEGMENTATION
   Champions & Loyal Customers -> retain via loyalty program.
   At Risk -> win-back offer within 30 days.
   Hibernating -> deep-discount re-engagement campaign.

6. COHORT RETENTION
   Retention drops sharply after Month 1 across all cohorts.
   Action: Introduce a post-purchase email sequence (Months 1-3)
   with personalised product recommendations.

=========================================================
""")


# In[ ]:




