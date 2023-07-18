
sh script.sh &&
echo "Train finished"

rm -f ./model/ShuttleNetWeight/prediction.csv &&
python generator.py ./model/ShuttleNetWeight &&
echo "Generation finished"

mv ./model/ShuttleNetWeight/prediction.csv ../../input/dataset/prediction.csv
python evaluation.py &&
tail -1 record_score.csv
