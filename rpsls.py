#coding:gbk
"""
��һ��С��Ŀ��Rock-paper-scissors-lizard-Spock
���ߣ����˷�
���ڣ�2020.11.24
"""

import random

# ����Ϊ�����Ϸ����Ҫ�õ����Զ��庯��

def name_to_number(name):
    """
    ����Ϸ�����Ӧ����ͬ������
    """
    if name=="ʯͷ":
     return 0
    elif name=="ʷ����":
     return 1
    elif name=="ֽ":
     return 2
    elif name=="����":
     return 3
    elif name=="����":
     return 4

    # ʹ��if/elif/else��佫����Ϸ�����Ӧ����ͬ������
    # ��Ҫ���Ƿ��ؽ��


     #��дִ�д���,������ɺ�passɾ��


def number_to_name(number):
    """
    ������ (0, 1, 2, 3, or 4)��Ӧ����Ϸ�Ĳ�ͬ����
    """
    if number==0:
        return "ʯͷ"
    elif number==1:
        return "ʷ����"
    elif number==2:
        return "ֽ"
    elif number==3:
        return "����"
    elif number==4:
        return "����"


    # ʹ��if/elif/else��佫��ͬ��������Ӧ����Ϸ�Ĳ�ͬ����
    # ��Ҫ���Ƿ��ؽ��

     #��дִ�д���,������ɺ�passɾ��


def rpsls(player_choice):
    """
    �û�����������һ��ѡ�񣬸���RPSLS��Ϸ��������Ļ�������Ӧ�Ľ��

    """
    if player_choice in ("ʯͷ","����","ֽ","ʷ����","����"):
        print("--------")
        print("����ѡ��Ϊ:%s"%player_choice)
        player_choice = name_to_number(choice_name)
        comp_number = random.randrange(0, 4)
        comp_choice = number_to_name(comp_number)
        print("������ѡ��Ϊ:%s"%comp_choice)
        if comp_number - player_choice == 1 or comp_number - player_choice == 2 or comp_number - player_choice <= -3:
            return print("����Ӯ��")
        elif comp_number == player_choice:
            return print("���ͻ���ѡ��һ����")
        elif comp_number - player_choice == -1 or comp_number - player_choice == -2 or comp_number - player_choice >= 3:
            return print("��Ӯ��")
    else:
        print("Error: No Correct Name")

    # ���"-------- "���зָ�
    # ��ʾ�û�������ʾ���û�ͨ�����̽��Լ�����Ϸѡ��������룬�������player_choice

    # ����name_to_number()�������û�����Ϸѡ�����ת��Ϊ��Ӧ���������������player_choice_number

    # ����random.randrange()�Զ�����0-4֮��������������Ϊ��������ѡ�����Ϸ���󣬴������comp_number

    # ����number_to_name()����������������������ת��Ϊ��Ӧ����Ϸ����

    # ����Ļ����ʾ�����ѡ����������

    # ����if/elif/else ��䣬����RPSLS������û�ѡ��ͼ����ѡ������жϣ�������Ļ����ʾ�жϽ��

    # ����û��ͼ����ѡ��һ��������ʾ�����ͼ��������һ���ء�������û���ʤ������ʾ����Ӯ�ˡ�����֮����ʾ�������Ӯ�ˡ�

     #����������ʾ��дִ�д��룬������ɺ�ɾ����pass


# �Գ�����в���
print("��ӭʹ��RPSLS��Ϸ")
print("----------------")
print("����������ѡ��:")
choice_name=input()
rpsls(choice_name)


