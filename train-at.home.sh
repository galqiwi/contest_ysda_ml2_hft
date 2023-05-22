rsync -azvh --delete -e "ssh -p 4545" . galqiwi.ru:/home/galqiwi/.contest_ysda_tmp &&
ssh -p 4545 galqiwi.ru 'bash -c "source /home/galqiwi/.virtualenvs/contest_ysda/bin/activate && cd /home/galqiwi/.contest_ysda_tmp && export PYTHONPATH=/home/galqiwi/.contest_ysda_tmp && python3 ./randomly_sample_hyperparameters.py"' &&
rsync -azvh --delete -e "ssh -p 4545" galqiwi.ru:/home/galqiwi/.contest_ysda_tmp/model ./
rsync -azvh --delete -e "ssh -p 4545" galqiwi.ru:/home/galqiwi/.contest_ysda_tmp/feature_importance.csv ./
