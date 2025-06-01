# This file reads data from three csvs (NCHS_-_Leading_Causes_of_Death__United_States.csv; historical_state_population_by_year.csv,
# and Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv; these files must be in the same
# directory) and transfers it to an SQL database using SQLite. Key data is then extracted from each and included into a fourth
# sql database.

# Author: Elizabeth Javor elizabethjavor@proton.me
# May 2025

###################################################################

import sqlite3
import csv
import os
from helperfunctions import *


# Takes in the name of a csv file and the name of the table it will be written to (abr), along with
# sqlite connection info, and creates a table in the current database with same info as csv.
def csvtosqldb(csvfile, abr, cur, con):
    if not os.path.exists(csvfile):
        raise FileNotFoundError(f"{csvfile} not found in current directory.")
    with open(csvfile, mode="r") as file:
        firstline = True
        data = []
        reader = csv.reader(file)
        for line in reader:
            sqlstr = ""
            tablestr = ""
            n = len(line)
            if firstline is True:
                for i, heading in enumerate(line):
                    newheading = sqlify(heading)
                    sqlstr = sqlstr + newheading
                    tablestr = tablestr + newheading + " TEXT"
                    if i < (n - 1):
                        sqlstr = sqlstr + ","
                        tablestr = tablestr + ","
                sqlstr = "(" + sqlstr + ")"
                tablestr = "(" + tablestr + ")"
                cur.execute(f"CREATE TABLE {abr} {tablestr}")
                firstline = False
            else:
                data.append(tuple(line))
        valuestr = "VALUES(" + ",".join(["?"] * n) + ")"
        cur.executemany(f"INSERT INTO {abr} {sqlstr} {valuestr}", data)
        con.commit()
#Creates database that holds various statistics vs deaths per million from some cause by merging previous sql databases
#created in main. 
#Statlist: list of the description/reference for each statistic
#labeltuples: markers for each datapoint, in form (State, Year) e.g. (Alabama, 2011)
#cause: Cause of death to be referenced in cause of death file
#cur, con: cursor and connection for SQLite

def tablemaker(cause, cur, con):
    labeltuples = cur.execute("SELECT DISTINCT State, Year FROM death").fetchall()
    statlist = [
        "Percent of adults aged 18 years and older who have an overweight classification",
        "Percent of adults aged 18 years and older who have obesity",
        "Percent of adults who engage in no leisure-time physical activity",
        "Percent of adults who engage in muscle-strengthening activities on 2 or more days a week",
        "Percent of adults who achieve at least 150 minutes a week of moderate-intensity aerobic physical activity or 75 minutes a week of vigorous-intensity aerobic activity (or an equivalent combination)",
        "Percent of adults who achieve at least 150 minutes a week of moderate-intensity aerobic physical activity or 75 minutes a week of vigorous-intensity aerobic physical activity and engage in muscle-strengthening activities on 2 or more days a week",
        "Percent of adults who achieve at least 300 minutes a week of moderate-intensity aerobic physical activity or 150 minutes a week of vigorous-intensity aerobic activity (or an equivalent combination)",
        "Percent of adults who report consuming fruit less than one time daily",
        "Percent of adults who report consuming vegetables less than one time daily",
    ]
    #Above is all the distinct questions in the cause of death CSV, ordered in a way that makes sense to me.
    tabname = sqlify(cause)
    cur.execute(f"DROP TABLE IF EXISTS {tabname}")
    creationstatement = f"CREATE TABLE {tabname} (id INTEGER PRIMARY KEY, State TEXT, Year INT, Deaths_Per_Million INT"
    for i, stat in enumerate(statlist):
        creationstatement += f", Stat{i} FLOAT"
    creationstatement += ")"
    cur.execute(creationstatement)
    for label in labeltuples:
        cur.execute(
            f"""
            INSERT INTO {tabname} (State, Year, Deaths_Per_Million)
            VALUES (?, ?, (
                SELECT (Deaths * 1000000 / population)
                FROM death
                JOIN pop ON (death.State = pop.State AND death.Year = pop.Year)
                WHERE death.State = ? AND death.Year = ? AND Cause_Name = ?
            ))
            """,
            (label[0], label[1], label[0], label[1], cause),
        )
        for i, stat in enumerate(statlist):
            cur.execute(
                f"""
                UPDATE {tabname}
                SET Stat{i} = (
                    SELECT Data_Value FROM physical
                    WHERE Question = ? AND LocationDesc = ? AND YearStart = ?
                )
                WHERE State = ? AND Year = ?
                """,
                (stat, label[0], label[1], label[0], label[1]),
            )
    con.commit()
    #Uncomment to use nonechecker
    #nonechecker(statlist, cause, cur, con)
    



#Sets up SQL databases from CSV files, then merges their information in a separate database.#
def main():
    con = sqlite3.connect("health.db")
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS death")
    cur.execute("DROP TABLE IF EXISTS pop")
    cur.execute("DROP TABLE IF EXISTS physical")
    csvtosqldb("data/NCHS_-_Leading_Causes_of_Death__United_States.csv", "death", cur, con)
    csvtosqldb(
        "data/Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv",
        "physical",
        cur,
        con,
    )
    csvtosqldb("data/historical_state_population_by_year.csv", "pop", cur, con)
    cur.executescript("""
    DELETE FROM physical WHERE StratificationID1 != 'OVERALL' OR (YearStart > 2017 OR YearStart < 2011);
    DELETE FROM death WHERE (Year > 2017 OR Year < 2011) OR State = 'United States';
    DELETE FROM pop WHERE Year > 2017 OR Year < 2011;
    CREATE INDEX IF NOT EXISTS idx_death_state_year ON death(State, Year);
    CREATE INDEX IF NOT EXISTS idx_physical_location_year ON physical(LocationDesc, YearStart);
    """)
    # The csv containing population has state abbreviations instead of names; below (until #####) updates them to names.
    statedict = dict(
        cur.execute(
            "SELECT DISTINCT LocationAbbr, LocationDesc FROM physical"
        ).fetchall()
    )
    for stateab in statedict.keys():
        cur.execute(
            "UPDATE pop SET State = "
            + quote(statedict[stateab])
            + " WHERE State = "
            + quote(stateab)
        )
    ######
    tablemaker("Diabetes", cur, con)
    tablemaker("Heart disease", cur, con)
    con.close()


if __name__ == "__main__":
    main()
