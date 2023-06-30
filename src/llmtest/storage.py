from llmtest import contants


def insert_data(name, question, gpt_answer, iwx_answer, answer_helpful=False):
    import mysql.connector
    db = mysql.connector.connect(
        host=contants.MYSQL_HOST,
        user=contants.MYSQL_USER,
        password=contants.MYSQL_PASSWD,
        database=contants.MYSQL_DB
    )
    cursor = db.cursor()
    val = (name, question, iwx_answer, gpt_answer)
    sql = """INSERT INTO chatbot_test
  (name, question, iwx_answer, gpt_answer) VALUES (%s, %s, %s, %s)"""
    cursor.execute(sql, val)
    db.commit()
    cursor.close()
    db.close()
