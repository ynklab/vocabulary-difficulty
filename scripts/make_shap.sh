# Generate SHAP data for the explainable model:

python scripts/run_features.py -m xgbr --sdiff-l1 --cqp morphtran --trickiness no --final-predict  explainable --no-save --output-shap explainable_shap -S grouped

# Make individual (local) plots for the web page:

python scripts/shap_plots_test.py explainable_shap -g -i --xm 1 --format png --dpi 72 --left 0.2

# Make global plots for the paper:

python scripts/shap_plots_test.py explainable_shap -g    --xm 0.7 --xd 1 --figure-scale 0.55 -L 0.24 -R 0.97 -T 0.99 -B 0.17  --print-titles

# Make plots for "farmhouse" for the paper:

python scripts/shap_plots_test.py explainable_shap -g -i --xm 1 --xd 1  --figure-scale 0.55 -L 0.3 -R 0.97 -T 0.99 -B 0.17  --print-titles --ids 6796  --outdir explainable_shap/plots/farmhouse --ld 2

# L1=Chinese, En=farmhouse, POS=noun, L1=农庄住宅（农场主的房子）, Pred=-0.20, Target=0.14
# L1=German, En=farmhouse, POS=noun, L1=Bauernhaus, Pred=0.65, Target=0.94
# L1=Spanish, En=farmhouse, POS=noun, L1=alquería, Pred=-1.87, Target=-2.14
