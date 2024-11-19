# TripDurationNYC
![NYC Taxi](https://static01.nyt.com/images/2007/10/18/nyregion/taxi600.jpg?quality=75&auto=webp)
## Abstraction
Building a model that predicts the total duration of taxi trips in New York City. We are working on a dataset released by the New York City Taxi and Limousine Commission, which includes pickup time, geographic coordinates, number of passengers, and many other variables, and we will talk about the details during the report.
## conclusion 
*dataset does not contain missing values.

*fixing structure error
 - We modified pickup_datetime , dropoff_datetime from object to datetime so that we can perform some operations on it.
   
*Handle outlets
   - After reviewing the statistical summary of the data set, we noticed that there are some values that are not possible for some columns, for example :
         -trip_duration We found that some trips lasted for more than 22 hours, which raises doubts as it is an impossible value.
         -passenger_count Some flights had no passengers.
         -some trips that have many passengers (7 passengers)

*vendor id 
  -	vendor id has two value 1,2
  -	vendor ID 2 has made more trips than  vendor id 1 , vendor ID 2 made 54% of trips.
  -	The different vendors did not affect the average trip duration, so their values were very close
    
*Number of passengers 
  -	71% of  trips had 1 passenger and 14 % trips  had 2 passengers and 5 % of  trips had 3 passengers.
  -	There are two trips have 7 passengers.
  -	 some trips that have zero passenger (38 trips)
    
*Longitude and latitude 
  -	From longitude and latitude of start and end point canextract new features -> distance
  -	Distance has clearly outliers ,some trips have distance 771 kilo meters that is impossible. 
  -	Trip duration and distance have high correlation, can you profit from it in your model.
  - from distance and Time can compute 	Speed:
      -	Speed is computed by divide distance by trip duration. 
      -	Can not used speed on your model but can get sum intuitions about correlated it with other features.
      -	but can used to get some information like :
          - The number of trips increases at noon and in the evening than at any other time.
          -	The number of flights increases on Friday, Saturday, and Thursday of each week.
          -	The speed of vehicles increased between night and late at night.
          -	The speed of vehicles increased on Sunday and Saturday
          -	The speed of vehicles increased in the months of 5 and 6
          - The speed of vehicles increased on holiday days than on other days.

*store_and_fwd_flag:
  -	90 %  of the trip not a store and forward trip.
### used external dataset (The Open-Source Routing Machine (OSRM) dataset)

+There are many columns that can help him predict like total_distance , total_travel_time , number_of_steps 

+You can also verify this by merging with the main dataset by column ID.

+Total distance :
  -	total_distance has negative skew
  -	most distance is between 0 and 5000 metre
+total travel time:
  -	total travel time has negative skew.
  -	most total travel time is between 0 and 16 minutes.
+number_of_steps:
  -	number_of_steps have negative skew.
  -	most number_of_steps is between 1.5 and 2.5 steps.

### used external dataset (Weather data in New York City - 2016)
+The dip in trip volume is attributed to a blizzard that hit NYC, causing record-breaking snowfall and significantly impacting traffic patterns.

+Subsequent periods of snowfall, even if less intense, do not have as substantial an effect on the number of taxi trips or their velocities.

+This could indicate that passengers were more likely to travel shorter distances on snowy days.

## modeling Result

| model  | Train R2-score | validation R2-score |
| :---         |     :---:      |          ---: |
| Ridge | 0.7298 |0.7311 |
| Random forest	  | 0.8351l  |0.7972 |
| Content XGBRegressor  | 0.8108  |0.8056 |

Source:
Kaggle : https://www.kaggle.com/c/nyc-taxi-trip-duration/data






