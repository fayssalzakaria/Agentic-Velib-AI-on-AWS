-- Synthèse journalière par station (dates en heure de Paris)
select
    date,
    station_id,
    arg_max(name, ingested_at)          as name,
    count(*)                            as n_snapshots,
    round(avg(numbikesavailable), 2)    as avg_bikes,
    min(numbikesavailable)              as min_bikes,
    max(numbikesavailable)              as max_bikes,
    sum(is_empty::int)                  as snapshots_empty,
    sum(is_full::int)                   as snapshots_full,
    round(avg(is_empty::int), 4)        as pct_empty
from {{ ref('stg_station_status') }}
group by date, station_id
