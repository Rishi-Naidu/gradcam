param(
    [string]$VideoPath="data/videos/my360.mp4",
    [string]$OutDir="outputs",
    [string]$Backbone="resnet50",
    [bool]$Pretrained=$true,
    [int]$TopK=1,
    [bool]$SavePng=$true
)
python -m src.run_gradcam --video $VideoPath --outdir $OutDir --backbone $Backbone --pretrained $Pretrained --topk $TopK --save-png $SavePng