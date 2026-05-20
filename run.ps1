param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsFromCaller
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python .\reproduce_kfpps.py @ArgsFromCaller