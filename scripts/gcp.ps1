# GCP operations for the project. Clusters delete themselves when idle.
#   .\scripts\gcp.ps1 setup
#   .\scripts\gcp.ps1 upload
#   .\scripts\gcp.ps1 cluster -Workers 7
#   .\scripts\gcp.ps1 resize -Workers 4
#   .\scripts\gcp.ps1 job01 [-Sample 0.05]
#   .\scripts\gcp.ps1 job02 [-Limit 2] [-RunName test]
#   .\scripts\gcp.ps1 sync
#   .\scripts\gcp.ps1 delete
param(
    [Parameter(Position = 0, Mandatory = $true)][string]$Action,
    [int]$Workers = 7,
    [string]$Machine = "n2-standard-2",
    [double]$Sample = 1.0,
    [int]$Limit = 0,
    [string]$RunName = "full",
    [string]$ClusterName = "regime-cluster",
    [string]$ReplayDate = "2023-10-30"
)

# gcloud writes progress to stderr, which "Stop" would treat as a failure in PowerShell 5.1.
$ErrorActionPreference = "Continue"
$Project = "big-data-hw0-509017"
$Region = "us-central1"
$Bucket = "gs://$Project-regime"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Packages = "hmmlearn==0.3.3,google-cloud-pubsub==2.27.1,gcsfs==2024.10.0"

function Submit-Spark([string]$Script, [string[]]$JobArgs, [string]$Properties = "") {
    & $Python (Join-Path $Root "scripts\package_code.py") | Out-Null
    $base = @("dataproc", "jobs", "submit", "pyspark", (Join-Path $Root "spark_jobs\$Script"),
        "--cluster=$ClusterName", "--region=$Region", "--project=$Project",
        "--py-files=$(Join-Path $Root 'dist\regime.zip')")
    if ($Properties) { $base += "--properties=$Properties" }
    # gcloud.ps1 would swallow the "--" separator, so call the .cmd wrapper.
    & gcloud.cmd @base -- @JobArgs
}

switch ($Action) {
    "setup" {
        gcloud services enable pubsub.googleapis.com --project $Project
        gcloud storage buckets create $Bucket --project $Project --location $Region
        bq --project_id=$Project mk --dataset --location=$Region regime
        gcloud pubsub topics create regime-bars --project $Project
        gcloud pubsub subscriptions create regime-bars-sub --topic regime-bars --project $Project --message-retention-duration=1h
    }
    "upload" {
        gcloud storage cp --recursive (Join-Path $Root "data\raw\bars") "$Bucket/raw/"
        gcloud storage cp (Join-Path $Root "data\raw\macro.parquet") "$Bucket/raw/macro.parquet"
    }
    "cluster" {
        gcloud dataproc clusters create $ClusterName --project $Project --region $Region `
            --image-version 2.2-debian12 --master-machine-type $Machine --master-boot-disk-size 100 `
            --worker-machine-type $Machine --worker-boot-disk-size 100 --num-workers $Workers `
            --public-ip-address --max-idle 20m --max-age 8h `
            --properties "^#^dataproc:pip.packages=$Packages#spark:spark.sql.execution.arrow.pyspark.enabled=true"
    }
    "resize" { gcloud dataproc clusters update $ClusterName --region $Region --project $Project --num-workers $Workers }
    "job01" { Submit-Spark "01_bars_to_daily.py" @("--bucket", $Bucket, "--sample-fraction", "$Sample") }
    "job02" { Submit-Spark "02_run_sweep.py" @("--bucket", $Bucket, "--limit-configs", "$Limit", "--run-name", $RunName) `
            "spark.executorEnv.OMP_NUM_THREADS=1,spark.executorEnv.OPENBLAS_NUM_THREADS=1" }
    "job03" { Submit-Spark "03_stream_regime.py" @("--bucket", $Bucket, "--project", $Project, "--replay-date", $ReplayDate) }
    "replay" { Submit-Spark "..\scripts\replay_bars.py" @("--date", $ReplayDate, "--project", $Project, "--bars-dir", "$Bucket/raw/bars") }
    "bigquery" {
        foreach ($kind in "metric", "quality", "stats") {
            bq --project_id=$Project load --replace --source_format=NEWLINE_DELIMITED_JSON "regime.sweep_$kind" `
                "$Bucket/results/$RunName/raw/kind=$kind/*.json" "payload:STRING"
        }
        bq --project_id=$Project load --replace --source_format=NEWLINE_DELIMITED_JSON --ignore_unknown_values regime.job_stats `
            "$Bucket/system/job_stats/*.json" "payload:STRING,job:STRING,raw_rows:INTEGER,clean_rows:INTEGER,daily_rows:INTEGER,seconds:FLOAT,executors:STRING,sample_fraction:FLOAT"
        bq --project_id=$Project load --replace --source_format=PARQUET regime.streaming_batches "$Bucket/results/streaming/$ReplayDate/*.parquet"
    }
    "sync" {
        $target = Join-Path $Root "results\runs"
        New-Item -ItemType Directory -Force -Path $target | Out-Null
        gcloud storage cp --recursive "$Bucket/results/*" $target
        gcloud storage cp --recursive "$Bucket/system" $target
        gcloud storage cp --recursive "$Bucket/curated" (Join-Path $Root "data")
    }
    "delete" { gcloud dataproc clusters delete $ClusterName --region $Region --project $Project --quiet }
    default { throw "Unknown action: $Action" }
}
