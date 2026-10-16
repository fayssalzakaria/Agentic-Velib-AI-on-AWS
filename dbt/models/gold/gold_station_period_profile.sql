-- Profil compact : station x (semaine / week-end) x période de la journée.
-- ~1 500 stations x 2 x 4 = ~12 000 lignes : assez petit pour construire un
-- document RAG par station et l'indexer dans Qdrant depuis Streamlit.
with hourly as (
    select
        *,
        case when is_weekend then 'weekend' else 'semaine' end as day_type,
        case
            when hour between 6 and 11  then 'matin'
            when hour between 12 and 17 then 'apres-midi'
            when hour between 18 and 21 then 'soir'
            else 'nuit'
        end as period
    from {{ ref('gold_station_hourly_profile') }}
)

select
    station_id,
    arg_max(name, last_seen)                                         as name,
    day_type,
    period,
    sum(n_obs)                                                       as n_obs,
    round(sum(avg_bikes * n_obs) / sum(n_obs), 2)                    as avg_bikes,
    round(sum(avg_ebike * n_obs) / sum(n_obs), 2)                    as avg_ebike,
    round(sum(avg_docks * n_obs) / sum(n_obs), 2)                    as avg_docks,
    round(sum(pct_empty * n_obs) / sum(n_obs), 4)                    as pct_empty,
    round(sum(pct_full * n_obs) / sum(n_obs), 4)                     as pct_full,
    round(
        sum(avg_bike_ratio * n_obs)
        / nullif(sum(case when avg_bike_ratio is not null then n_obs end), 0),
    4)                                                               as avg_bike_ratio,
    min(first_seen)                                                  as first_seen,
    max(last_seen)                                                   as last_seen
from hourly
group by station_id, day_type, period
