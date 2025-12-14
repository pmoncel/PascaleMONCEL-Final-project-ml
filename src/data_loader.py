
import kagglehub
import pandas as pd

# Charge le dataset et retourne un dataframe pandas
def load_data():
    data_path = kagglehub.dataset_download(
        "albertobircoci/support-ticket-priority-dataset-50k"
    )
    return pd.read_csv(data_path + "/support_tickets.csv")
