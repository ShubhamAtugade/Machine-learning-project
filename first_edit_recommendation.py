# language: Python 
# library: Pandas, Seaborn, Matplotlib, Scikit-surprise

# **Loading and analyzing the dataset for better result**

#Step 1: Importing necessary libraries
import pandas as pd

#Step 2: Load the dataset into jupyter notebook
data = pd.read_csv('OnlineRetail.csv')

#Step 3: Viewing 5 rows of data
data.head(5)

#Step 4: getting the dataset information and the total count
data.info()

#Step 5: Checking for empty columns in dataset
data.isnull().sum()

#Step 6: Dropping the empty columns in dataset
data.dropna(inplace=True)

#Step 7: Dropping the duplicates values from dataset
data.drop_duplicates(inplace=True)

data.info()

# +
#Step 8: displaying the unique product and customer count

#Here we use "stock code" for unique products and "CustomerID" for unique customers.
print(f'Number of unique products : {data['StockCode'].nunique()}')
print(f'Number of unique customers : {data['CustomerID'].nunique()}')
# -

#Displaying the top 10 products from the dataset
top_products = data['Description'].value_counts().head(10)
print (top_products)

# **Lets visualize using plot for better analyzing**

#Step 1: importing necessary library
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt

#Step 2: Creating and displaying the pivot table to find the total quantity of each product bought by each customer
total_product_quantity = data.pivot_table(
    index = 'CustomerID',
    columns = 'Description',
    values = 'Quantity',
    aggfunc = 'sum',
    fill_value = 0
)
display(total_product_quantity)

# +
#Plot the distribution of the number of products bought by each customer using histogram

plt.figure(figsize=(10, 6))
sns.histplot(data=total_product_quantity.sum(axis=1), kde=True)
plt.title('Distribution of the Number of Products Bought by Each Customer')
plt.xlabel('Number of Products')
plt.ylabel('Frequency')
plt.show()
# -

#Step 3: Identifying Globally Popular Products
globally_popular_products = data['Description'].value_counts().head(10)
print("Globally Popular Products:")
print(globally_popular_products)

#Plotting globally popular products
plt.figure(figsize=(10, 6))
sns.barplot(x=globally_popular_products.values, y=globally_popular_products.index)
plt.title('Top 10 Globally Popular Products')
plt.xlabel('Quantity Sold')
plt.ylabel('Product Description')
plt.show()

#Step 4: Identify Country-wise Popular Products
country_popular_products = data.groupby('Country')['Description'].value_counts().groupby(level=0).nlargest(10).reset_index(level=0, drop=True).reset_index()
print("Country-wise Popular Products:")
print(country_popular_products.head(20)) 

#Plot country-wise popular products for a specific country
country = input('Enter country name: ')  # enter country from dataset
popular_products = country_popular_products[country_popular_products['Country'] == country]
plt.figure(figsize=(10, 6))
sns.barplot(x=popular_products['Description'].value_counts().values, 
            y=popular_products['Description'].value_counts().index )
plt.title(f'Top 10 Popular Products in {country}')
plt.xlabel('Quantity Sold')
plt.ylabel('Product Description')
plt.show()

# +
#Step 5: Identifying Month-wise Popular Products

#Convert InvoiceDate to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
# -

#Extract month and year from InvoiceDate
data['MonthYear'] = data['InvoiceDate'].dt.to_period('M')

month_popular_products = data.groupby('MonthYear')['Description'].value_counts().groupby(level=0).nlargest(10).reset_index(level=0, drop=True).reset_index()
print("Month-wise Popular Products:")
print(month_popular_products.head(20))

#Plot month-wise popular products for a specific month
specific_month = input('Enter year and month (yyyy-mm): ')
monthly_popular_products = month_popular_products[month_popular_products['MonthYear'] == specific_month]
plt.figure(figsize=(10, 6))
sns.barplot(x=monthly_popular_products['Description'].value_counts().values, 
            y=monthly_popular_products['Description'].value_counts().index)
plt.title(f'Top 10 Popular Products in {specific_month}')
plt.xlabel('Quantity Sold')
plt.ylabel('Product Description')
plt.show()

# **Getting product recommedations for customer**

# Here I use collaborative filtering using the surprise library for getting recommendations

#Step 1: Importing necessary library
from surprise import Dataset, Reader, SVD 
from surprise.model_selection import cross_validate 

#Creating a Reader object and specifying the rating scale
reader = Reader(rating_scale=(0, data['Quantity'].max()))

#Creating the dataset from the pandas dataframe
data_for_surprise = Dataset.load_from_df(data[['CustomerID', 'StockCode', 'Quantity']], reader)

#Using the SVD algorithm for collaborative filtering
algo = SVD()

#Evaluating the algorithm with cross-validation
cross_validate(algo, data_for_surprise, measures=['RMSE', 'MAE'], cv=5, verbose=True)

#Training the model on the entire dataset
trainset = data_for_surprise.build_full_trainset()
algo.fit(trainset)


# +
#Function to get top n recommendations for a given customer

def top_recommendations(customer_id, n=15):

    customer_id = float(customer_id)
    
    #list of all products
    all_products = data['Description'].unique()
    
    #list of products the customer has already bought
    purchased_products = data[data['CustomerID'] == customer_id]['Description'].unique()
    
    #list of products the customer has not bought yet 
    products_to_predict = [product_description for product_description in all_products if product_description not in purchased_products] 
    
    # Predict the ratings for all products the customer has not bought yet
    predictions = [algo.predict(customer_id, product_description) for product_description in products_to_predict]
    
    # Sort the predictions by estimated rating
    predictions.sort(key=lambda x: x.est)
        
    # top N recommendations
    top_recommendations = [pred.iid for pred in predictions[:n]]
    
    return top_recommendations


# +
customer_id = input('Enter the Customer ID')
top_product_recommendations = top_recommendations(customer_id, n=5)
print(f'\nTop 10 recommendated products for {customer_id}:')

for product in top_product_recommendations:
    print(product)
    
