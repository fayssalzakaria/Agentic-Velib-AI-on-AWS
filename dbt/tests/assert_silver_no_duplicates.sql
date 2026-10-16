-- Le MERGE silver doit garantir une seule ligne par (snapshot, station)
select snapshot_id, station_id, count(*) as n
from {{ ref('stg_station_status') }}
group by snapshot_id, station_id
having count(*) > 1
