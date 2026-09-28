from src.logging_utils import write_prediction_log, read_prediction_logs

def test_logging_round_trip_isolated(tmp_path):
    log_dir=tmp_path/"test_logs"
    write_prediction_log(log_dir, {"country":"Australia","total_prediction":123})
    records=read_prediction_logs(log_dir)
    assert len(records)==1
    assert records[0]["country"]=="Australia"
