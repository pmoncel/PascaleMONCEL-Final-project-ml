# data_loader.py
# Charge le dataset et retourne un dataframe pandas
import kagglehub
import pandas as pd


def load_data():
    """
    Charge le dataset et retourne un dataframe pandas
    """
    data_path = kagglehub.dataset_download(
        "albertobircoci/support-ticket-priority-dataset-50k"
    )
    return pd.read_csv(data_path + "/support_tickets.csv")
