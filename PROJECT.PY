import pandas as pd
import numpy as np
import requests
import mysql.connector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA




api_key_ow = '499f1171108e6c7bf613c0537f7dcf7c'           # open weather api key
api_key_tt = "YKPHuZ74pdng8wmhRnlxECWA6r2DQu36"           # tom tom api key

    
df = pd.read_csv('FINAL_DS.csv')



conn = mysql.connector.connect(
    host="localhost",        
    user="root",     
    password="GAUTAM", 
    database="ml_project" 
)


cursor = conn.cursor()



def EDA():
    from ydata_profiling import ProfileReport
    prof = ProfileReport(df, correlations={"pearson": {"calculate": True}, "spearman": {"calculate": True}})
    prof.to_file(output_file="output_Final.html")



def Handling_missing_values():
    
    numeric_columns = ['Delivery_person_Age', 'temperature','humidity','precipitation','Distance (km)']
    categorical_columns = ['Type_of_order', 'Type_of_vehicle','weather_description','Traffic_Level']
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    for col in categorical_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df.to_csv('Filled_missing_value.csv', index=False)

    print("Missing values have been filled for specified columns and saved in the same file.")



def Ordinal_Encoding():
    df=pd.read_csv('Filled_missing_value.csv')
    column_to_encode = 'Traffic_Level'
    category_order = [['Very Low', 'Low', 'Moderate','High','Very High']]
    encoder = OrdinalEncoder(categories=category_order)
    df[[column_to_encode]] = encoder.fit_transform(df[[column_to_encode]])
    df.to_csv('Ordinal_encoded.csv', index=False)
    print("Ordinal encoding applied and saved to 'your_file.csv'")


def Nominal_encoding():
    df=pd.read_csv('Ordinal_encoded.csv')
    columns_to_encode = ['Type_of_order', 'Type_of_vehicle','weather_description']
    label_encoder = LabelEncoder()
    for col in columns_to_encode:
        df[col] = label_encoder.fit_transform(df[col])

    output_path = 'Encoded.csv'
    df.to_csv(output_path, index=False)
    print(f"Encoded CSV saved to {output_path}")


  




def Xgboost_without_pca():
    # Load the dataset and select specific columns
    file_path = 'Encoded.csv'  # Replace with your actual file path
    selected_columns = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Distance (km)',
                         'temperature', 'precipitation', 'humidity', 'Traffic_Level', 'Travel_Time_Minutes']
    data = pd.read_csv(file_path, usecols=selected_columns)

    # Clean the target column by converting non-numeric values to NaN and dropping them
    data['Travel_Time_Minutes'] = pd.to_numeric(data['Travel_Time_Minutes'], errors='coerce')
    data = data.dropna(subset=['Travel_Time_Minutes'])

    # Define feature columns and target column
    target_column = 'Travel_Time_Minutes'
    X = data.drop(columns=[target_column])
    y = data[target_column].astype(float)

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Initialize the XGBoost model for regression
    model = XGBRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)  # Fit the model to the training data

    # Now you can make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Display Model Performance Metrics Table
    metrics_df = pd.DataFrame({
        'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'R-squared (R²)'],
        'Value': [mae, mse, rmse, r2]
    })
    print(metrics_df)




    # Actual vs. Predicted Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='purple', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)  # Diagonal line
    plt.xlabel('Actual Travel Time')
    plt.ylabel('Predicted Travel Time')
    plt.title('Actual vs. Predicted Travel Time')
    plt.show()

    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, color='blue', alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Travel Time')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

    # Error Distribution Plot
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='orange')
    plt.xlabel('Prediction Error')
    plt.title('Error Distribution')
    plt.show()

    # Predicted vs Actual Delivery Time Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(y_test, color='blue', label='Actual', kde=True, alpha=0.5)
    sns.histplot(y_pred, color='red', label='Predicted', kde=True, alpha=0.5)
    plt.xlabel('Delivery Time (Minutes)')
    plt.title('Predicted vs Actual Delivery Time Distribution')
    plt.legend()
    plt.show()



    # Return the trained model to be used later
    return model




def Xgboost_with_pca():


    # Load the dataset and select specific columns
    file_path = 'Encoded.csv'  # Replace with your actual file path
    selected_columns = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Distance (km)',
                        'temperature', 'precipitation', 'humidity', 'Traffic_Level', 'Travel_Time_Minutes']
    data = pd.read_csv(file_path, usecols=selected_columns)

    # Display first few rows to understand the structure
    print(data.head())

    # Clean the target column by converting non-numeric values to NaN and dropping them
    data['Travel_Time_Minutes'] = pd.to_numeric(data['Travel_Time_Minutes'], errors='coerce')
    data = data.dropna(subset=['Travel_Time_Minutes'])

    # Define feature columns and target column
    target_column = 'Travel_Time_Minutes'
    X = data.drop(columns=[target_column])
    y = data[target_column].astype(float)

    # Handle NaN values in the feature columns
    X = X.fillna(X.median())

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply PCA
    pca = PCA(n_components=7)
    x_pca = pca.fit_transform(X_train)
    x_test_pca = pca.transform(X_test)  # Transform the test set as well

    # Initialize the XGBoost model for regression
    model = XGBRegressor(n_estimators=100, random_state=42)

    # Train the model with the PCA-transformed training data
    model.fit(x_pca, y_train)

    # Make predictions on the PCA-transformed test data
    y_pred = model.predict(x_test_pca)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Display Model Performance Metrics Table
    metrics_df = pd.DataFrame({
        'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'R-squared (R²)'],
        'Value': [mae, mse, rmse, r2]
    })
    print(metrics_df)

    # Feature Importance Plot
    feature_importances = pd.DataFrame(model.feature_importances_,
                                    index=[f'PCA Component {i+1}' for i in range(x_pca.shape[1])],
                                    columns=['Importance']).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances.index, feature_importances['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()  # Invert y-axis for a descending order
    plt.show()






    # Actual vs. Predicted Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='purple', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)  # Diagonal line
    plt.xlabel('Actual Travel Time')
    plt.ylabel('Predicted Travel Time')
    plt.title('Actual vs. Predicted Travel Time')
    plt.show()

    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, color='blue', alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Travel Time')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

    # Error Distribution Plot
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='orange')
    plt.xlabel('Prediction Error')
    plt.title('Error Distribution')
    plt.show()

    # Predicted vs Actual Delivery Time Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(y_test, color='blue', label='Actual', kde=True, alpha=0.5)
    sns.histplot(y_pred, color='red', label='Predicted', kde=True, alpha=0.5)
    plt.xlabel('Delivery Time (Minutes)')
    plt.title('Predicted vs Actual Delivery Time Distribution')
    plt.legend()
    plt.show()

    return model



def add_restaurants(lat,lon):
    id=int(input("Enter the Restaurant_id : "))
    name=input("Enter the Restaurants Name : ")
    r_lat=lat
    r_lon=lon
    sql = "INSERT INTO restaurants (restaurant_id, restaurant_name,longitude,latitude) VALUES (%s, %s, %s, %s)"
    values = (id, name, r_lon, r_lat)
    cursor.execute(sql, values)
    conn.commit()  





def add_destiantion(lat,lon):
    id=int(input("Enter the Destination_id : "))
    name=input("Enter the Destination Address : ")
    r_lat=lat
    r_lon=lon
    sql = "INSERT INTO destination (destination_id , Destination_addre,longitude,latitude) VALUES (%s, %s, %s, %s)"
    values = (id, name, r_lon, r_lat)
    cursor.execute(sql, values)
    conn.commit()  







def get_weather_data(api_key,lat,lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key_ow}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        precipitation = data.get('rain', {}).get('1h', 0)  
        weather_desc = data['weather'][0]['description']
        return temp, humidity, precipitation, weather_desc
    else:
        return None, None, None, None
    






def get_traffic_info(lat1, lon1, lat2, lon2):
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{lat1},{lon1}:{lat2},{lon2}/json"
    params = {
        'traffic': 'true',
        'key': api_key_tt,
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        try:
            # Access the route summary
            summary = data['routes'][0]['summary']
            travel_time = summary['travelTimeInSeconds'] / 60  # Convert seconds to minutes
            
            # Categorize traffic level based on travel time
            if travel_time < 10:
                traffic_level = "Very Low"
            elif travel_time < 20:
                traffic_level = "Low"
            elif travel_time < 30:
                traffic_level = "Moderate"
            elif travel_time < 45:
                traffic_level = "High"
            else:
                traffic_level = "Very High"
            
            return travel_time, traffic_level
        except (IndexError, KeyError) as e:
            print("Error accessing data:", e)
            return "N/A", "N/A"
    else:
        print(f"Failed to fetch data, status code: {response.status_code}")
        return "N/A", "N/A"



'''
def get_traffic_info(lat1, lon1, lat2, lon2, api_key_tt):
    # URL for the TomTom routing API
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{lat1},{lon1}:{lat2},{lon2}/json"
    params = {
        'traffic': 'true',
        'key': api_key_tt,
    }
    
    # Sending request to the API
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        try:
            # Access the route summary
            summary = data['routes'][0]['summary']
            
            # Get normal travel time (without traffic) and travel time considering traffic
            normal_travel_time = summary['baseTimeInSeconds'] / 60  # Convert seconds to minutes
            travel_time_with_traffic = summary['travelTimeInSeconds'] / 60  # Convert seconds to minutes
            
            # Calculate the difference in time
            time_difference = travel_time_with_traffic - normal_travel_time
            
            # Categorize traffic level based on the time difference
            if time_difference <= 0:
                traffic_level = "Very Low"
            elif time_difference <= 5:
                traffic_level = "Low"
            elif time_difference <= 15:
                traffic_level = "Moderate"
            elif time_difference <= 30:
                traffic_level = "High"
            else:
                traffic_level = "Very High"
            
            return normal_travel_time, travel_time_with_traffic, time_difference, traffic_level
        except (IndexError, KeyError) as e:
            print("Error accessing data:", e)
            return "N/A", "N/A", "N/A", "N/A"
    else:
        print(f"Failed to fetch data, status code: {response.status_code}")
        return "N/A", "N/A", "N/A", "N/A"

'''

def get_tomtom_distance(api_key, origin, destination):
    # TomTom API URL for routing service
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{origin}:{destination}/json?key={api_key}"

    # Send the GET request to TomTom API
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise an HTTPError for bad responses (4xx or 5xx)

        # Try to parse the response as JSON
        data = response.json()

        # Check if the response contains valid data
        if 'routes' in data and len(data['routes']) > 0:
            # Extract the distance from the response (in meters)
            distance_meters = data['routes'][0]['summary']['lengthInMeters']
            distance_km = distance_meters / 1000  # Convert meters to kilometers
            return distance_km
        else:
            print(f"Error: No route found for origin: {origin}, destination: {destination}")
            return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as err:
        print(f"Error occurred: {err}")
    except ValueError as json_err:
        print(f"JSON decode error: {json_err}")

    return None






def Fetch_values():
    cursor = conn.cursor()
    age = int(input("Enter the age : "))
    rating = float(input("Enter the rating : "))
    r_id = int(input("Enter the Restaurant_id: "))
    

    query = f"SELECT longitude, latitude FROM restaurants WHERE restaurant_id = {r_id}"
    cursor.execute(query)
    result = cursor.fetchone()

    if result:
        restaurant_lon = result[0]  
        restaurant_lat = result[1]

    d_id = int(input("Enter the Destination id : "))
    query = f"SELECT longitude, latitude FROM destination WHERE destination_id = {d_id}"
    cursor.execute(query)
    result = cursor.fetchone()

    if result:
        delivery_lon = result[0]  
        delivery_lat = result[1]

    origin = f"{restaurant_lat},{restaurant_lon}"
    destination = f"{delivery_lat},{delivery_lon}"
    distance = get_tomtom_distance(api_key_tt, origin, destination)

    temp, humidity, precipitation, weather_desc = get_weather_data(api_key_ow, delivery_lat, delivery_lon)
    travel_time, traffic_level = get_traffic_info(restaurant_lat, restaurant_lon, delivery_lat, delivery_lon)
    
    return age, rating, distance, temp, precipitation,humidity, traffic_level,weather_desc


def Predict_xgboost(model,age, rating, distance, temp, precipitation,humidity, traffic_level):
    # Ordinal encoding for 'Traffic_Level'
    category_order = [['Very Low', 'Low', 'Moderate', 'High', 'Very High']]
    encoder = OrdinalEncoder(categories=category_order)
    traffic_encoded = encoder.fit_transform([[traffic_level]])[0][0]


    new_input = pd.DataFrame({
        'Delivery_person_Age': [age],
        'Delivery_person_Ratings': [rating],
        'temperature': [temp],
        'humidity': [humidity],
        'precipitation': [precipitation],
        'Traffic_Level': [traffic_encoded],
        'Distance (km)': [distance]
        
        
          # Use the encoded value
    })

    # Make prediction
    prediction = model.predict(new_input)
    print(f"Predicted Travel Time for the input: {prediction[0]:.2f} minutes")







#EDA()
#Handling_missing_values()
#Ordinal_Encoding()
#Nominal_encoding()




#add_restaurants(30.340807698299233, 76.43478427438586)
#add_destiantion(30.351334617754073, 76.36462736281393)



#age, rating, distance, temp, precipitation,humidity, traffic_level,weather_desc=Fetch_values()



# print(age)
# print(rating)
# print(distance)
# print(temp)
# print(precipitation)
# print(humidity)
# print(traffic_level)
# print(weather_desc)
# print(o_type)


#model=Xgboost_without_pca()
#model=Xgboost_with_pca()







#Predict_xgboost(model,age, rating, distance, temp, precipitation,humidity, traffic_level)  






cursor.close()
conn.close()


