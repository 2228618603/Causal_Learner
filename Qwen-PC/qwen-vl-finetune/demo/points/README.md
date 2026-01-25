# Demo point-cloud asset

GitHub rejects single files larger than 100MB in normal Git history. The raw
`scene0000_01_vh_clean.ply` (~105MB) is therefore not tracked. The repo keeps a
compressed copy instead: `scene0000_01_vh_clean.ply.gz`.

Restore the original `.ply` after cloning:

```bash
cd Qwen-PC/qwen-vl-finetune/demo/points
gzip -dk scene0000_01_vh_clean.ply.gz
# If your gzip doesn't support `-k`, use:
#   gunzip -k scene0000_01_vh_clean.ply.gz
# or (will remove the .gz):
#   gzip -d scene0000_01_vh_clean.ply.gz
```
