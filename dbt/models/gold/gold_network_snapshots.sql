-- Indicateurs réseau à chaque snapshot (série temporelle globale)
select
    snapshot_id,
    any_value(ingested_at)              as ingested_at,
    any_value(date)                     as date,
    any_value(hour)                     as hour,
    any_value(weekday)                  as weekday,
    any_value(ingested_by)              as ingested_by,
    count(*)                            as n_stations,
    sum(numbikesavailable)              as total_bikes,
    sum(ebike)                          as total_ebike,
    sum(mechanical)                     as total_mechanical,
    sum(numdocksavailable)              as total_docks,
    sum(is_empty::int)                  as empty_stations,
    sum(is_full::int)                   as full_stations,
    round(avg(bike_ratio), 4)           as avg_bike_ratio
from {{ ref('stg_station_status') }}
group by snapshot_id
