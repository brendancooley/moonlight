import streamlit as st
import pandas as pd

from moonlight.valuations import ValuationModel


@st.cache
def load_valuations():
    vm = ValuationModel()
    return pd.read_csv(vm.output_path)


if __name__ == "__main__":
    st.title("Ottoneu Draft Helper")
    value_scalar = st.number_input("Value Scale:", min_value=0., max_value=100., value=1.)

    data = load_valuations()
    print(data["league_id"].unique())
    league_id = st.selectbox(label="Select League:", options=data["league_id"].unique())

    data_league = data.loc[data["league_id"] == league_id]
    teams = data_league["teamname"].unique()
    n_teams = len(teams)
    data_league["Value_Scaled"] = data_league["Value"] * value_scalar
    data_league["Surplus"] = data_league["Value_Scaled"] - data_league["Salary"]
    data_league_nfa = data_league.loc[data_league['teamname'] != "free_agent"]
    st.write(f"Total dollars commited: ${int(data_league['Salary'].sum())}")
    st.write(f"Dollars available: ${400*n_teams - int(data_league['Salary'].sum())}")
    st.write(f"Implied rostered value: "
             f"${int(data_league_nfa['Value_Scaled'].sum())}")
    team_summary = data_league_nfa.groupby('teamname').agg({'Salary': 'sum',
                                                            'Surplus': 'sum',
                                                            'Value_Scaled': 'sum'}).sort_values('Surplus',
                                                                                                ascending=False)
    st.dataframe(team_summary)

    st.header("Team Inspection")
    team = st.radio("Select Team:", teams)
    data_team = (data_league.loc[data_league["teamname"] == team].drop(columns=["teamname", "league_id", "mlbamid"])
                 .sort_values("Value_Scaled", ascending=False))
    left_cols = ["firstname", "lastname", "Salary", "Value_Scaled", "Surplus"]
    data_team = data_team[left_cols + [col for col in data_team.columns if col not in left_cols]]
    st.dataframe(data_team)




