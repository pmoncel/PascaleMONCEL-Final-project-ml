import pandas as pd


def apply_feature_engineering(data):
    """
    Nouvelles features créées :
    1. impact_ratio : Ratio customers_affected / org_users
       Impact d'un incident / taille de l'organisation

    2. is_problematic_client : past_90d_incidents > threshold ?
       => Clients ayant un historique d'incidents élevé
       (threshold = 75e % de past_90d_incidents)

    3. description_category : Description_length devient court/moyen/long

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    data = data.copy()

    print("\n" + "="*25)
    print("FEATURE ENGINEERING APPLIQUÉ")
    print("="*25)

    # 1 : impact_ratio
    data['impact_ratio'] = data['customers_affected'] / data['org_users'].replace(0, 1)
    print(f"✓ impact_ratio créé (customers_affected / org_users)")
    print(f"  - Min: {data['impact_ratio'].min():.4f}, Max: {data['impact_ratio'].max():.4f}, Moyenne: {data['impact_ratio'].mean():.4f}")

    # 2 : is_problematic_client
    incidents_threshold = data['past_90d_incidents'].quantile(0.75)
    data['is_problematic_client'] = (data['past_90d_incidents'] > incidents_threshold).astype(int)
    print(f"\n✓ is_problematic_client créé (past_90d_incidents > {incidents_threshold})")
    print(f"  - Clients problématiques : {data['is_problematic_client'].sum()} / {len(data)} ({100*data['is_problematic_client'].mean():.2f}%)")

    # 3 : description_category
    p33 = data['description_length'].quantile(0.33)
    p66 = data['description_length'].quantile(0.66)

    def categorize_description(length):
        if length <= p33:
            return 'court'
        elif length <= p66:
            return 'moyen'
        else:
            return 'long'

    data['description_category'] = data['description_length'].apply(categorize_description)
    print(f"\n✓ description_category créé (court ≤{p33:.0f}, moyen ≤{p66:.0f}, long >{p66:.0f})")
    print(f"  - Distribution : {data['description_category'].value_counts().to_dict()}")

    print("="*25)
    print(f"3 nouvelles features créées")
    print("="*25 + "\n")

    return data
