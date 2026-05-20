# Open track
############

## finetuned_llms: best finetuned LLM ensemble

python scripts/run_features.py --sdiff-l1 --cqp morphtran -f --final-train finetuned_llms_plus
python scripts/run_features.py --sdiff-l1 --cqp morphtran -f --final-predict finetuned_llms_plus --track open

# CV PERFORMANCE: 
# python scripts/run_features.py --cv -q  --sdiff-l1 --cqp morphtran -f


## finetuned_llms_plus: + explainable (incl. trickiness)

python scripts/run_features.py --sdiff-l1 --cqp morphtran -f --final-train finetuned_llms_plus
python scripts/run_features.py --sdiff-l1 --cqp morphtran -f --final-predict finetuned_llms_plus --track open

# CV PERFORMANCE: 
# python scripts/run_features.py --cv -q  --sdiff-l1 --cqp morphtran -f


## open_max: above + difficulty prompting

python scripts/run_features.py --sdiff-l1 --cqp morphtran -D -f --final-train open_max
python scripts/run_features.py --sdiff-l1 --cqp morphtran -D -f --final-predict open_max --track open

# CV PERFORMANCE: 
# python scripts/run_features.py --cv -q --sdiff-l1 --cqp morphtran -D -f


# Closed track
##############

## explainable: traditional features + prompting

python scripts/run_features.py -m xgbr --sdiff-l1 --cqp morphtran --trickiness no --final-train explainable
python scripts/run_features.py -m xgbr --sdiff-l1 --cqp morphtran --trickiness no --final-predict explainable --track closed

# CV PERFORMANCE: 
# python scripts/run_features.py -m xgbr --cv -q --sdiff-l1 --cqp morphtran --trickiness no


# With trickiness (not in the final submission):
# python scripts/run_features.py -m xgbr --sdiff-l1 --cqp morphtran --final-train explainable
# python scripts/run_features.py -m xgbr --sdiff-l1 --cqp morphtran --final-predict explainable --track closed
# # CV PERFORMANCE: 
# python scripts/run_features.py -m xgbr --cv -q --sdiff-l1 --cqp morphtran

## traditional: traditional features

python scripts/run_features.py -m xgbr --no-prompting --final-train traditional
python scripts/run_features.py -m xgbr --no-prompting --final-predict traditional --track closed

# CV PERFORMANCE: 
#python scripts/run_features.py -m xgbr --cv -q --no-prompting


## closed_max: extra features, even with small effect, finetuned mmbert (closed)

python scripts/run_features.py -m xgbr --sdiff-l1 --cqp morphtran calque --bnc spoken written --gn cnc imag fam aoa -G --os -f --fc mmbert-base-closed --trickiness no --final-train closed_max
python scripts/run_features.py -m xgbr --sdiff-l1 --cqp morphtran calque --bnc spoken written --gn cnc imag fam aoa -G --os -f --fc mmbert-base-closed --trickiness no --final-predict closed_max --track closed

# CV PERFORMANCE: 
# python scripts/run_features.py --cv -q -m xgbr --sdiff-l1 --cqp morphtran calque --bnc spoken written --gn cnc imag fam aoa -G --os -f --fc mmbert-base-closed --trickiness no 

# With trickiness:
# python scripts/run_features.py -m xgbr --sdiff-l1 --cqp morphtran calque --bnc spoken written --gn cnc imag fam aoa -G --os -f --fc mmbert-base-closed --final-train closed_max
# python scripts/run_features.py -m xgbr --sdiff-l1 --cqp morphtran calque --bnc spoken written --gn cnc imag fam aoa -G --os -f --fc mmbert-base-closed --final-predict closed_max --track closed
# # CV PERFORMANCE: 
# python scripts/run_features.py --cv -q -m xgbr --sdiff-l1 --cqp morphtran calque --bnc spoken written --gn cnc imag fam aoa -G --os -f --fc mmbert-base-closed 


# Closed track ABLATIONS
########################


python scripts/run_features.py -m lr --sdiff-l1 --cqp morphtran --trickiness no --final-train explainable_lr
python scripts/run_features.py -m lr --sdiff-l1 --cqp morphtran --trickiness no --final-predict explainable_lr --track closed --submission-directory submission-extra

python scripts/run_features.py -m xgbr --sdiff-l1 --cqp morphtran --trickiness no --temperatures all=1 --final-train explainable_temp1
python scripts/run_features.py -m xgbr --sdiff-l1 --cqp morphtran --trickiness no --temperatures all=1 --final-predict explainable_temp1 --track closed --submission-directory submission-extra

python scripts/run_features.py -m xgbr --sdiff-l1 --cqp morphtran --trickiness no --temperatures all=0.001 --final-train explainable_temp0
python scripts/run_features.py -m xgbr --sdiff-l1 --cqp morphtran --trickiness no --temperatures all=0.001 --final-predict explainable_temp0 --track closed --submission-directory submission-extra

python scripts/run_features.py -m lr --sdiff-l1 --cqp morphtran --trickiness no --temperatures all=1 --final-train explainable_temp1_lr
python scripts/run_features.py -m lr --sdiff-l1 --cqp morphtran --trickiness no --temperatures all=1 --final-predict explainable_temp1_lr --track closed --submission-directory submission-extra

# python scripts/run_features.py -m xgbr --sdiff-l1 --cqp morphtran --trickiness auto --difficulty --final-train explainable+t+d
# python scripts/run_features.py -m xgbr --sdiff-l1 --cqp morphtran --trickiness auto --difficulty --final-predict explainable+t+d --track closed --submission-directory submission-extra

# Attic
#######

# Other experiments/comparison:

# - just_ministral
#     
#     ```bash
#     python scripts/run_features.py --no-default-features -f --fc ministral-3-14b --final-train just_ministral
#     python scripts/run_features.py --no-default-features -f --fc ministral-3-14b --final-predict just_ministral --track open
#     
#     # CV PERFORMANCE: 
#     python scripts/run_features.py --cv -q --no-default-features -f --fc ministral-3-14b
#     cn: features: 0.921
#     de: features: 0.905
#     es: features: 0.914
#     
#     # TEST PERFORMANCE
#     python scripts/run_features.py --no-default-features -f --fc ministral-3-14b --final-eval just_ministral --track open
#     
#     L1	rmse	pearson	r2
#     cn	0.683	0.915	0.835 (Glite: 0.660)
#     de	0.777	0.902	0.813 (Glite: 0.764)
#     es	0.795	0.907	0.822 (Glite: 0.754)
#     
#     ```
#     
# - just_glm
#     
#     ```bash
#     python scripts/run_features.py --no-default-features -f --fc glm-4-32b --final-train just_glm
#     python scripts/run_features.py --no-default-features -f --fc glm-4-32b --final-predict just_glm --track open
#     
#     # CV PERFORMANCE: 
#     python scripts/run_features.py --cv -q --no-default-features -f --fc glm-4-32b
#     cn: features: 0.915
#     de: features: 0.897
#     es: features: 0.906
#     
#     # TEST PERFORMANCE
#     python scripts/run_features.py --no-default-features -f --fc glm-4-32b --final-eval just_glm --track open
#     
#     L1	rmse	pearson	r2
#     cn	0.671	0.918	0.840 (Glite: 0.660)
#     de	0.760	0.907	0.821 (Glite: 0.764)
#     es	0.801	0.906	0.819 (Glite: 0.754)
#     ```