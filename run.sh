 for i in 0 0 0 1 1 1; do
    # Set the CUDA visible devices and run the Python script in the background, redirecting output to a log file
    CUDA_VISIBLE_DEVICES=$i nohup python -u run_ourmatch.py > console_$i.txt &
    # Pause for 10 seconds before starting the next run
    sleep 10
done