-- Trip metrics by pickup hour (NYC taxi sample dataset)
SELECT
  hour(tpep_pickup_datetime)        AS pickup_hour,
  count(*)                          AS total_trips,
  round(avg(trip_distance), 2)      AS avg_distance_mi,
  round(avg(fare_amount), 2)        AS avg_fare_usd
FROM samples.nyctaxi.trips
GROUP BY hour(tpep_pickup_datetime)
ORDER BY pickup_hour
