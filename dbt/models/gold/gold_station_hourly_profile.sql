-- Profil type d'une station : comportement moyen par jour de semaine et par heure.
-- C'est la table que le RAG indexe : des tendances, pas des lignes brutes.
with recent as (
    select *
    from {{ ref('stg_station_status') }}
    where ingested_at >= current_timestamp - interval '{{ var("profile_days") }} days'
)

select
    station_id,
    arg_max(name, ingested_at)                      as name,
    weekday,
    hour,
    weekday in ('Saturday', 'Sunday')               as is_weekend,
    count(*)                                        as n_obs,
    round(avg(numbikesavailable), 2)                as avg_bikes,
    round(avg(ebike), 2)                            as avg_ebike,
    round(avg(mechanical), 2)                       as avg_mechanical,
    round(avg(numdocksavailable), 2)                as avg_docks,
    round(avg(is_empty::int), 4)                    as pct_empty,
    round(avg(is_full::int), 4)                     as pct_full,
    round(avg(bike_ratio), 4)                       as avg_bike_ratio,
    min(ingested_at)                                as first_seen,
    max(ingested_at)                                as last_seen
from recent
group by station_id, weekday, hour
