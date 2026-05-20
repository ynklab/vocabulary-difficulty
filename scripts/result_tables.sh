python scripts/mean_prediction_files.py --pred-files \
	predictions/finetuned_llm/test/mar20calibrated-GLM-4-32B-Base-allinone-lr1x--zai-org--GLM-4-32B-Base-0414.csv \
	predictions/finetuned_llm/test/mar20calibrated-allinone--mistralai--Ministral-3-14B-Base-2512-lr1x-ep4.csv \
	predictions/finetuned_llm/test/mar20calibrated-Qwen2.5-32B-allinone-lr1p5x--Qwen--Qwen2.5-32B.csv \
	--output-path predictions/finetuned_llm/test/14b-to-32b-llms-mean.csv


python scripts/mean_prediction_files.py --pred-files \
	predictions/finetuned_llm/test/apr16calibrated-Qwen2.5-14B-allinone-lr1x-full--Qwen--Qwen2.5-14B.csv \
	predictions/finetuned_llm/test/apr16calibrated-glm-4-9b-allinone-lr1x-full--zai-org--glm-4-9b.csv \
	predictions/finetuned_llm/test/mar20calibrated-allinone--mistralai--Ministral-3-14B-Base-2512-lr1x-ep4.csv \
	--output-path predictions/finetuned_llm/test/9b-to-14b-llms-mean.csv


python scripts/mean_prediction_files.py --pred-files \
	predictions/finetuned_llm/test/apr16calibrated-glm-4-9b-allinone-lr1x-full--zai-org--glm-4-9b.csv \
	predictions/finetuned_llm/test/apr16calibrated-Qwen2.5-7B-allinone-lr1x-full--Qwen--Qwen2.5-7B.csv \
	predictions/finetuned_llm/test/apr16calibrated-Ministral-3-8B-Base-allinone-lr1x-full--mistralai--Ministral-3-8B-Base-2512.csv \
	--output-path predictions/finetuned_llm/test/7b-to-9b-llms-mean.csv

python scripts/block_reverse_test.py -vw --esb 69 --cnb 95 --deb 108 -o predictions/finetuned_llm/test/block-reversed.csv

python scripts/evaluate_prediction_files_table.py --pred-files \
	open:open_max \
	open:finetuned_llms_plus \
	open:finetuned_llms \
	14b-to-32b-llms-mean.csv \
	mar20calibrated-GLM-4-32B-Base-allinone-lr1x--zai-org--GLM-4-32B-Base-0414.csv \
	mar20calibrated-Qwen2.5-32B-allinone-lr1p5x--Qwen--Qwen2.5-32B.csv \
	mar20calibrated-allinone--mistralai--Ministral-3-14B-Base-2512-lr1x-ep4.csv \
	9b-to-14b-llms-mean.csv \
	7b-to-9b-llms-mean.csv \
	--row-labels \
	'\narrowtt{open\_max}' \
	'\narrowtt{finetuned\_llms\_plus}' \
	'\narrowtt{finetuned\_llms}' \
	/ \
	'$\le$32B LLM Average' \
	'- GLM-4-32B' \
	'- Qwen2.5-32B' \
	'- Ministral-3-14B' \
	'$\le$14B LLM Average' \
	'$\le$9B LLM Average' \
	/ \
	-b open \
	--block-reversed 'Statistical Optimum' \
	--label-id open-models-ensembles -x \
	--output-csv results/open-models-ensembles.csv --output-tex results/result_tables.tex
		
	
python scripts/evaluate_prediction_files_table.py --pred-files \
	mar20calibrated-allinone--mistralai--Ministral-3-14B-Base-2512-lr1x-ep4.csv \
	apr16calibrated-Ministral-3-14B-abl-langs--mistralai--Ministral-3-14B-Base-2512.csv \
	apr21calibrated-Ministral-3-14B-abl-ool--mistralai--Ministral-3-14B-Base-2512.csv \
	apr16calibrated-Ministral-3-14B-abl-prompt--mistralai--Ministral-3-14B-Base-2512.csv \
	apr16calibrated-Ministral-3-14B-abl-ce-prob--mistralai--Ministral-3-14B-Base-2512.csv \
	apr16calibrated-Ministral-3-14B-abl-ce-top--mistralai--Ministral-3-14B-Base-2512.csv \
	--row-labels \
	'\makebox[0pt][l]{Ours (Ministral-3-14B)}' \
	'- single language' \
	'- out-of-language' \
	'- short prompt' \
	'- standard loss' \
	'- std.\ loss \& inference' \
	--first-label 'Method (Base Model)' --label-id llm-ablation -x \
	--output-csv results/llm-ablation.csv --output-tex results/result_tables.tex -a

#	apr16calibrated-Ministral-3-14B-abl-raft--mistralai--Ministral-3-14B-Base-2512.csv \
#	M-3-14B-RAFT \

python scripts/evaluate_prediction_files_table.py --pred-files \
	apr18-mmbert-ep16-cnesde-allinone--jhu-clsp--mmBERT-base.csv \
	mar26-mmbert-ep16-cnesde--jhu-clsp--mmBERT-base.csv \
	apr22-mmbert-ep16-cnesde-abl-ool-fd--jhu-clsp--mmBERT-base.csv	\
	apr18-mmbert-ep16-cnesde-abl-prompt--jhu-clsp--mmBERT-base.csv \
	apr18-mmbert-ep16-cnesde-abl-ce-prob--jhu-clsp--mmBERT-base.csv \
	apr18-mmbert-ep16-cnesde-abl-ce-top--jhu-clsp--mmBERT-base.csv \
	apr19-xlmr-base-openxx-ep5-allinone--xlm-roberta-base.csv \
	apr19-xlmr-large-openxx-ep5-allinone--xlm-roberta-large.csv \
	apr19-mmbert-ep16-cnesde-allinone-abl-reg--jhu-clsp--mmBERT-base.csv \
	apr19-mmbert-ep16-cnesde-abl-reg--jhu-clsp--mmBERT-base.csv \
	--row-labels \
	'Ours (mmBERT-b)' \
	'- single language' \
	'- out-of-language' \
	'- short prompt' \
	'- standard loss' \
	'- std.\ loss \& inference' \
	/ \
	'Regression (XLMR-b)' \
	'Regression (XLMR-l)' \
	'Reg.\ (mmBERT-b)' \
	'%- single language' \
	--first-label 'Method (Base Model)' --label-id mlm-ablation -x \
	--output-csv results/mlm-ablation.csv --output-tex results/result_tables.tex -a

# apr18-mmbert-ep16-cnesde-abl-raft--jhu-clsp--mmBERT-base.csv



python scripts/evaluate_prediction_files_table.py --pred-files \
	closed:closed_max \
	closed:explainable \
	closed:traditional \
	submission-extra:closed:explainable_temp0 \
	submission-extra:closed:explainable_lr \
	--row-labels \
	'\texttt{closed\_max}' \
	'\texttt{explainable}' \
	'\texttt{traditional}' \
	/ \
	'\texttt{exp.}: std. inference' \
	'\texttt{exp.}: lin. regression' \
	/ \
	-b closed \
	--first-label System --label-id closed-models -x \
	--output-csv results/closed_models.csv --output-tex results/result_tables.tex -a

# 	submission-extra:closed:explainable_temp1 \
# 	submission-extra:closed:explainable_temp1_lr \
	
# 	submission+trickiness:closed:explainable-with-trickiness \
# 	submission-extra:closed:explainable+t+d \
# 
# 	'\texttt{explainable} + trickiness' \
# 	'\texttt{explainable} + trick.\ + diff.' \


python scripts/evaluate_prediction_files_table.py --pred-files \
	mar20calibrated-allinone--mistralai--Ministral-3-14B-Base-2512-lr1x-ep4.csv \
	apr25calibrated-Ministral-3-14B-probspace--mistralai--Ministral-3-14B-Base-2512.csv \
	--row-labels \
	'Ministral-3-14B (logit)' \
	'Ministral-3-14B (prob)' \
	--label-id probspace \
	--output-csv results/probspace.csv --output-tex results/result_tables.tex -a

# 	predictions/finetuned_llm/test/apr27calibrated-Ministral-3-3B-Base-allinone-lr1x-full--mistralai--Ministral-3-3B-Base-2512.csv \
# 	predictions/finetuned_llm/test/apr27calibrated-Qwen2.5-3B-allinone-lr1x-full--Qwen--Qwen2.5-3B.csv \
# 	\
# 	'Ministral-3-3B' \
# 	'Qwen2.5-3B' \
# 	/ \
python scripts/evaluate_prediction_files_table.py --pred-files \
	predictions/finetuned_llm/test/mar20calibrated-GLM-4-32B-Base-allinone-lr1x--zai-org--GLM-4-32B-Base-0414.csv \
	predictions/finetuned_llm/test/mar20calibrated-Qwen2.5-32B-allinone-lr1p5x--Qwen--Qwen2.5-32B.csv \
	\
	predictions/finetuned_llm/test/mar20calibrated-allinone--mistralai--Ministral-3-14B-Base-2512-lr1x-ep4.csv \
	predictions/finetuned_llm/test/apr16calibrated-Qwen2.5-14B-allinone-lr1x-full--Qwen--Qwen2.5-14B.csv \
	\
	predictions/finetuned_llm/test/apr16calibrated-glm-4-9b-allinone-lr1x-full--zai-org--glm-4-9b.csv \
	predictions/finetuned_llm/test/apr16calibrated-Ministral-3-8B-Base-allinone-lr1x-full--mistralai--Ministral-3-8B-Base-2512.csv \
	predictions/finetuned_llm/test/apr16calibrated-Qwen2.5-7B-allinone-lr1x-full--Qwen--Qwen2.5-7B.csv \
	\
	predictions/finetuned_llm/test/apr27calibrated-Qwen2.5-1.5B-allinone-lr1x-full--Qwen--Qwen2.5-1.5B.csv \
	\
	predictions/finetuned_llm/test/apr27calibrated-Qwen2.5-0.5B-allinone-lr1x-full--Qwen--Qwen2.5-0.5B.csv \
	predictions/finetuned_llm/test/apr27-xlmr-large-mmbertsetup-ep5-allinone--xlm-roberta-large.csv \
	\
	predictions/finetuned_llm/test/apr18-mmbert-ep16-cnesde-allinone--jhu-clsp--mmBERT-base.csv \
	predictions/finetuned_llm/test/apr27-xlmr-base-mmbertsetup-ep5-allinone--xlm-roberta-base.csv \
	\
	--row-labels \
	'GLM-4-32B' \
	'Qwen2.5-32B' \
	/ \
	'Ministral-3-14B' \
	'Qwen2.5-14B' \
	/ \
	'GLM-4-9B' \
	'Ministral-3-8B' \
	'Qwen2.5-7B' \
	/ \
	'Qwen2.5-1.5B' \
	/ \
	'Qwen2.5-0.5B' \
	'XLMR-l (550B)' \
	/ \
	'mmBERT-b (307M)' \
	'XLMR-b (270B)' \
	--first-label 'Base Model' \
	--label-id all-llms -x \
	--output-csv results/open-models-ensembles.csv --output-tex results/result_tables.tex -a

