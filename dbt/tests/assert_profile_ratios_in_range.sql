-- Les pourcentages du profil doivent rester entre 0 et 1
select *
from {{ ref('gold_station_hourly_profile') }}
where pct_empty not between 0 and 1
   or pct_full not between 0 and 1
   or (avg_bike_ratio is not null and avg_bike_ratio not between 0 and 1)
