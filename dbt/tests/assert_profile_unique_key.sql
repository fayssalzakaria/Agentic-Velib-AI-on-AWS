-- Une seule ligne par (station, jour de semaine, heure)
select station_id, weekday, hour, count(*) as n
from {{ ref('gold_station_hourly_profile') }}
group by station_id, weekday, hour
having count(*) > 1
