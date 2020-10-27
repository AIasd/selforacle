from typing import List

from eval_db.database import Database

INSERT_STATEMENT = "INSERT INTO settings ('id', 'agent', 'track') values (?,?,?);"


class Setting:
    def __init__(self, id: int, agent: str, track: str):
        self.id: int = id
        self.agent: str = agent
        self.track: str = track

    def get_folder_name(self):
        return self.track

    def insert_into_db(self, db: Database) -> None:
        db.cursor.execute(INSERT_STATEMENT,
                          (self.id, self.agent, self.track))


def get_all_settings(db: Database) -> List[Setting]:
    cursor = db.cursor.execute('select * from settings')
    var = cursor.fetchall()
    result = []
    for db_record in var:
        print(db_record)
        setting = Setting(id=db_record[0], agent=db_record[1], track=db_record[2])
        result.append(setting)
    return result
