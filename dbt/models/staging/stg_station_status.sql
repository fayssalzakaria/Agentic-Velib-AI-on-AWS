-- Table silver Delta Lake écrite par pipeline/lake.py
select *
from delta_scan('{{ env_var("LAKE_ROOT") }}/silver/station_status')
