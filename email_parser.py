import pandas as pd
import time
import ast
import csv
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag.stanford import StanfordNERTagger, StanfordPOSTagger
from nltk.parse.stanford import StanfordDependencyParser, StanfordParser

st = StanfordNERTagger('.stanford/stanford_ner/english.all.3class.distsim.crf.ser.gz', './stanford/stanford_ner/stanford-ner.jar')

parser = StanfordDependencyParser(
                        "./stanford/stanford_parser/stanford-parser.jar",
                        "./stanford/stanford_parser/stanford-parser-3.7.0-models.jar")

st_parser = StanfordParser("./stanford/stanford_parser/stanford-parser.jar",
                        "./stanford/stanford_parser/stanford-parser-3.7.0-models.jar")

st_pos = StanfordPOSTagger("./stanford/stanford_postagger/english-left3words-distsim.tagger",
                           path_to_jar="./stanford/stanford_postagger/stanford-postagger.jar")


class ReadWriteData:
    def __init__(self, read_path):
        self.read_path = read_path
        print "Data Reading and Writing class has been instantiated."

    def read_raw_data(self):
        data_frame = pd.read_csv(self.read_path)
        return data_frame

    def data_frame_to_csv(self, data_frame, path):
        data_frame.to_csv(path, encoding="utf-8")


class PreProcessing:
    def __init__(self, data_frame):
        self.data = data_frame

    def ___get_thread_ids(self):
        data = self.data
        all_thread_ids = data["thread-id"]
        unique_thread_ids = pd.unique(all_thread_ids)
        return unique_thread_ids

    def ___get_emails_for_thread_number(self, thread_number):
        data = self.data
        thread_data = data.loc[data["thread-id"] == thread_number]
        return thread_data

    def get_email_by_thread(self):
        thread_data_dictionary = {}
        thread_ids = self.___get_thread_ids()
        print "All unique thread-ids have been extracted."
        for thread_id in thread_ids:
            thread_data = self.___get_emails_for_thread_number(thread_id)
            thread_data_dictionary.update({thread_id: thread_data})
        print "All emails as per their thread-ids have been extracted."
        return thread_data_dictionary


class EmailQualityTagger:
    def __init__(self, email_dictionary):
        self.__email_dict = email_dictionary
        print "Email quality tagging based on thread-id count and content has been initiated."

    def tag_by_thread_id(self):
        good_thread_ids = []
        bad_thread_ids = []
        for key in self.__email_dict.keys():
            emails = self.__email_dict[key]["content"]
            unique_emails = pd.unique(emails)
            if len(unique_emails) > 1:
                good_thread_ids.append(key)
            else:
                bad_thread_ids.append(key)
        return good_thread_ids, bad_thread_ids

    def tag_by_content_and_make_dataframe(self, data_frame, good_mail_list_by_thread):
        good_mail_tag = "Good Mail"
        bad_mail_tag = "Bad Mail"
        data_frame_column = []
        for row in data_frame.iterrows():
            if row[0] % 1000 == 0:
                print row[0], "rows have been labelled."
                print "==========="
            email = row[1]["content"]
            email = str(email)
            email = email.replace("\r","")
            email = email.replace("\n"," ")
            email = email.replace("\t"," ")
            if email.find("---------------------- Forwarded by") != -1:
                data_frame_column.append(good_mail_tag)
            elif email.find("-----Original Message-----") != -1:
                data_frame_column.append(good_mail_tag)
            elif row[1]["thread-id"] in good_mail_list_by_thread:
                data_frame_column.append(good_mail_tag)
            else:
                data_frame_column.append(bad_mail_tag)
        data_frame["quality_tag"] = data_frame_column
        data_frame = data_frame.sort_values("thread-id")
        return data_frame


class EmailParser:
    def __init__(self, data_frame):
        self.data = data_frame

    def parse_forwarded_by_header(self, text):
        print "FORWARDED BY HEADER DATA EXTRACTION WAS CALLED"
        print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        # headers: To: From: Subject: cc:
        header_data_dictonary = {}
        email_body = self.extract_body_under_forwarded_by_header(text)
        header_data_dictonary.update({"email_body" : email_body})
        text = str(text)
        start_index = 0
        header_label_positions = []
        to_pos = text.find("To:")
        if to_pos > 0:
            header_label_positions.append(to_pos)
        cc_pos = text.find("cc:")
        if cc_pos > 0:
            header_label_positions.append(cc_pos)
        subject_pos = text.find("Subject:")
        if subject_pos > 0:
            header_label_positions.append(subject_pos)
        last_header = max(header_label_positions)
        header = text[start_index:last_header]
        split_header = header.splitlines()
        each_line = [x for x in split_header if len(x) > 2]  # to clean empty lines in between
        previous_line = ""
        to_header = header[header.find("To:"):header.find("cc:")]
        to_header = to_header.lstrip("To:")
        to_names = self.extract_name_from_labels(to_header)
        header_data_dictonary.update({"to_label": to_names})
        if header.find("Subject:") > 0:
            cc_header = header[header.find("cc:"):header.find("Subject:")]
            cc_header = cc_header.lstrip("cc:")
            cc_names = self.extract_name_from_labels(cc_header)
            header_data_dictonary.update({"cc_label": cc_names})
        else:
            cc_header = header[header.find("cc:"):]
            cc_header = cc_header.lstrip("cc:")
            cc_names = self.extract_name_from_labels(cc_header)
            header_data_dictonary.update({"cc_label": cc_names})
        for line in each_line:
            line = line.lstrip("\t")
            line = line.rstrip("\t")
            if line.startswith("From:"):
                from_header = line
                from_header.lstrip("From:")
                from_name = self.extract_name_from_labels(from_header)
                header_data_dictonary.update({"from_label": from_name})
            elif previous_line.endswith("------------"):
                from_header = line
                from_name = self.extract_name_from_labels(from_header)
                header_data_dictonary.update({"from_label": from_name})
            previous_line = line
        keys = ["to_label", "email_body", "from_label", "cc_label"]
        for key in keys:
            if key not in header_data_dictonary.keys():
                header_data_dictonary.update({key : []})
        return header_data_dictonary

    def parse_original_message_header(self, text):
        print "ORIGINAL MESSAGE HEADER DATA EXTRACTION WAS CALLED"
        print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        header_data_dictionary = {}
        header_start_index = text.find("From:")
        if header_start_index < 0:  # if there is no From: tag
            header_start_index = text.find("To:")  # will return -1 if it cant find it
        header_end_index = text.find("Subject:")
        end_of_subject_line = text[header_end_index:].find("\n")
        # find first \n after subject title. body will start after this
        header = text[header_start_index:header_end_index]
        header_portions = header.split("\n")
        for portion in header_portions:
            portion = portion.lstrip(">")
            portion = portion.lstrip("\t")
            portion = portion.lstrip("\r")
            portion = portion.lstrip(">      >      ")
            portion = portion.lstrip()
            if portion.startswith("To:"):
                # print "TO HEADER:", portion
                portion = portion.lstrip("To: ")
                names_list = self.get_data_from_original_subheaders(portion)
                header_data_dictionary.update({"to_label": names_list})
            elif portion.startswith("From:"):
                #print "FROM HEADER:", portion
                portion = portion.lstrip("From: ")
                names_list = self.get_data_from_original_subheaders(portion)
                header_data_dictionary.update({"from_label": names_list})
            elif portion.startswith("cc:") or portion.startswith("Cc"):
                # print "CC HEADER:", portion
                portion = portion.lstrip("cc: ")
                portion = portion.lstrip("Cc: ")
                names_list = self.get_data_from_original_subheaders(portion)
                header_data_dictionary.update({"cc_label": names_list})
            header_data_dictionary.update({"email_body": self.extract_body_under_forwarded_by_header(text)})
        keys = ["to_label", "email_body", "from_label", "cc_label"]
        for key in keys:
            if key not in header_data_dictionary.keys():
                header_data_dictionary.update({key: []})
        return header_data_dictionary

    def extract_body_under_forwarded_by_header(self, text):
        subj_pos = str(text).find("Subject:")
        cc_pos = str(text).find("cc:")
        # the last label would either be cc: or subject. to find that
        last_header = max([subj_pos, cc_pos])
        text_portion = text[last_header:]
        orig_message_pos = str(text_portion).find("-----Original Message-----")
        lines = text_portion.splitlines()
        #  first line will be of the last header of so find the length of it and use the text minus the length of this
        length = len(lines[0])
        if orig_message_pos > 0:
            body = text_portion[length:orig_message_pos]
        else:
            body = text_portion[length:]
        return body

    def extract_name_from_labels(self, text):
        names_list = []
        if len(text) > 1:
            recipients_list = text.split(",")  # to separate each entry if multiple recipients
            for recipient in recipients_list:
                recipient_name_start_index = 0
                recipient = recipient.lstrip("\t")
                recipient = recipient.lstrip()
                recipient_name_end_index = recipient.find("/")
                if recipient_name_end_index < 0:
                    recipient_name_end_index = recipient.find("@")  # "'Pallen@Enron.com'" <Pallen@Enron.com>
                recipient_name = recipient[recipient_name_start_index:recipient_name_end_index]
                names_list.append(recipient_name)
        return names_list

    def get_data_from_original_subheaders(self, text):
        names_list = []
        if len(text) > 1:
            recipients_list = text.split(";")  # original message has names like this:
            # To: surname, initial name; surname2, initial name2
            for recipient in recipients_list:
                if "@" not in recipient:
                    full_name = ""
                    # each recipient: surname, initial name
                    recipient_name_start_index = 0
                    recipient = recipient.lstrip("\t")
                    recipient = recipient.lstrip()
                    full_name = ' '.join(recipient.split(",")[::-1])
                    full_name.replace("\\r", " ")
                    full_name.replace("  ", " ")  # cuz there are multiple \r line se sath sath
                    names_list.append(full_name)
                else:
                    names = recipient.split("@")
                    names_list.append(names[0])
        return names_list

    def get_names_from_headerless_data(self, text):
        text = text.lstrip(">")
        text = text.lstrip("\t")
        text = text.lstrip("\r")
        text = text.lstrip(">      >      ")
        text = text.lstrip()
        to_pos = text.find("To:")
        from_pos = text.find("From:")
        if from_pos > 0 and from_pos < to_pos:
            text_before_header_less_labels = text[:from_pos]
        elif from_pos < 0 and to_pos > 0:
            text_before_header_less_labels = text[:to_pos]
        else:
            text_before_header_less_labels = ""
        return text_before_header_less_labels

    def parse_small_forwarded_by_header(self,text):
        print text
        to_pos = text.find("To:")
        whole_from_label = text[: to_pos]
        if whole_from_label.find("\"")>0:
            name_portion = whole_from_label[whole_from_label.find("\""):]
            name_end_pos = name_portion.find("\"")
            name = name_portion[:name_end_pos]
            print "======================================="
            print name
            return name

    def parse_email(self):
        useful_data = self.data.loc[self.data["quality_tag"] == "Good Mail"]
        print len(useful_data)
        new_data_frame = pd.DataFrame()
        message_ids_column = []
        thread_id_column = []
        to_name_list = []
        from_name_list = []
        full_content_list = []
        text_list = []
        count = 0
        for row in useful_data.iterrows():
            print count
            count += 1
            full_email = str(row[1]["content"])
            # print full_email
            org_pos = full_email.find("-----Original Message-----")
            fwd_pos = full_email.find("---------------------- Forwarded by")
            weird_fwd_pos = full_email.find("----- Forwarded by")
            if org_pos < 0 and fwd_pos < 0 and weird_fwd_pos < 0:  # means the email is just plain org_text
                print "CASE I !!!"
                message_ids_column.append(row[1]["Message-ID"])
                thread_id_column.append(row[1]["thread-id"])
                to_names = str(row[1]["ToNames"]) + "," + str(row[1]["cc"]) + "," + str(row[1]["bcc"])
                to_name_list.append(to_names)
                from_name_list.append(row[1]["FromNames"])
                full_content_list.append(row[1]["content"])
                text_list.append(full_email)
            elif fwd_pos < 0 and org_pos > 0:
                separate_text = full_email.split("-----Original Message-----")
                for text in separate_text:
                    if text.find("To:") > 0:
                        # could be a main header less email too.
                        body_before_headerless_labels = self.get_names_from_headerless_data(text)
                        if len(body_before_headerless_labels) > 1 and \
                                body_before_headerless_labels.find("---------------------------") < 0:
                            message_ids_column.append(row[1]["Message-ID"])
                            thread_id_column.append(row[1]["thread-id"])
                            to_names = str(row[1]["ToNames"]) + "," + str(row[1]["cc"]) + "," + str(row[1]["bcc"])
                            to_name_list.append(to_names)
                            from_name_list.append(row[1]["FromNames"])
                            full_content_list.append(row[1]["content"])
                            text_list.append(body_before_headerless_labels)
                        header_data = self.parse_original_message_header(text)
                        if header_data["email_body"].find("From:") > 0:
                           sub_org_text = header_data["email_body"][header_data["email_body"].find("From:"):]
                           sub_org_separated_texts = sub_org_text.split("From:")
                           for sub_text in sub_org_separated_texts:
                               if len(sub_text) > 1:
                                   lines = sub_text.splitlines()
                                   from_name_list.append(lines[0])
                                   sub_text_header_data = self.parse_original_message_header(sub_text)
                                   message_ids_column.append(row[1]["Message-ID"])
                                   thread_id_column.append(row[1]["thread-id"])
                                   full_content_list.append(row[1]["content"])
                                   if "cc_label" in sub_text_header_data.keys():
                                       to_names = sub_text_header_data["to_label"] + sub_text_header_data["cc_label"]
                                   else:
                                       to_names = sub_text_header_data["to_label"]
                                   to_name_list.append(to_names)
                                   text_list.append(sub_text_header_data["email_body"])
                           if "to_label" in header_data.keys():
                               message_ids_column.append(row[1]["Message-ID"])
                               thread_id_column.append(row[1]["thread-id"])
                               full_content_list.append(row[1]["content"])
                               if "cc_label" in header_data.keys():
                                   to_names = header_data["to_label"] + header_data["cc_label"]
                               else:
                                   to_names = header_data["to_label"]
                               to_name_list.append(to_names)
                               if 'from_label' in header_data.keys():
                                   from_name_list.append(header_data["from_label"])
                               else:
                                   from_name_list.append([])
                               text_list.append(header_data["email_body"][:header_data["email_body"].find("From:")])
                        if "to_label" in header_data.keys():
                            message_ids_column.append(row[1]["Message-ID"])
                            thread_id_column.append(row[1]["thread-id"])
                            full_content_list.append(row[1]["content"])
                            if "cc_label" in header_data.keys():
                                to_names = header_data["to_label"] + header_data["cc_label"]
                            else:
                                to_names = header_data["to_label"]
                            to_name_list.append(to_names)
                            if 'from_label' in header_data.keys():
                                from_name_list.append(header_data["from_label"])
                            else:
                                from_name_list.append([])
                            text_list.append(header_data["email_body"])
                    else:
                        message_ids_column.append(row[1]["Message-ID"])
                        thread_id_column.append(row[1]["thread-id"])
                        to_names = str(row[1]["ToNames"]) + "," + str(row[1]["cc"]) + "," + str(row[1]["bcc"])
                        to_name_list.append(to_names)
                        from_name_list.append(row[1]["FromNames"])
                        full_content_list.append(row[1]["content"])
                        text_list.append(text)
            elif fwd_pos > 0 and org_pos < 0:
                print full_email
                separate_text = full_email.split("---------------------- Forwarded by")
                for sub_mail in separate_text:
                    if len(sub_mail) > 1:
                        sub_main_data = self.parse_forwarded_by_header(sub_mail)
                        if sub_main_data["email_body"].find("Subject:") > 0:
                            print "d"
        new_data_frame["Message-Id"] = message_ids_column
        new_data_frame["Thread-Id"] = thread_id_column
        new_data_frame["ToNames"] = to_name_list
        new_data_frame["FromNames"] = from_name_list
        new_data_frame["EmailBody"] = text_list
        new_data_frame["FullEmailText"] = full_content_list
        return new_data_frame


class ApplyNLPTechiniques:
    def __init__(self, path):
        print "The csv file is being read."
        column_names = ['Message-Id','Thread-Id','ToNames','FromNames','EmailBody','FullEmailText']
        # self.data = pd.DataFrame.from_csv(path, encoding='utf-8')
        self.data = pd.read_csv(path, header=0, names=column_names, skiprows=1031, nrows=1005)
        print self.data

    def add_sender_name(self, text, name):
        #print "Name add sender name function got:", name
        name = name.replace("|Corp|Enron", "")
        rx = re.compile('[^a-zA-Z]')
        edited_name = rx.sub(' ', name).strip()
        #print "name the function will put in text:", edited_name
        words = text.split()
        count = 0
        replaced_text = ""
        for word in words:
            if word.lower() == "i":
                count += 1
                word = edited_name
            elif word.lower() == "i've":
                count += 1
                word = edited_name + " have"
            elif word.lower() == "i'll":
                count += 1
                word = edited_name + " will"
            elif word.lower() == "i'd":
                count += 1
                word = edited_name + " would"
            elif word.lower() == "i'm":
                count += 1
                word = edited_name + " am"
            replaced_text = replaced_text + " " + word
        return edited_name, replaced_text, count

    # you, you're, you'll, you'd, you've
    def add_receiver_name(self, text, receiver_name):
        all_edited_names = []
        each_name = receiver_name.split(",")
        #print "Name add receiver name function got:", receiver_name
        for name in each_name:
            if len(name) > 2:
                name = name.replace("|Corp|Enron","")
                rx = re.compile('[^a-zA-Z]')
                edited_name = rx.sub(' ', name).strip()
                all_edited_names.append(edited_name)
        #print "Name it will replace:", all_edited_names
        receiver_name = ','.join(all_edited_names)
        words = text.split()
        count = 0
        replaced_text = ""
        for word in words:
            if word.lower() == "you":
                count += 1
                word = receiver_name
            elif word.lower() == "you've":
                count += 1
                word = receiver_name + " have"
            elif word.lower() == "you'll":
                count += 1
                word = receiver_name + " will"
            elif word.lower() == "you'd":
                count += 1
                word = receiver_name + " would"
            elif word.lower() == "you're":
                count += 1
                word = receiver_name + " are"
            replaced_text = replaced_text + " " + word
        return receiver_name, replaced_text, count

    def change_text(self):
        data = self.data
        for row in data.iterrows():
            email = row[1]["content"]
            receiver_name = row[1]["ToNames"]
            sender_name = row[1]["FromNames"]
            changed_email = self.add_receiver_name(email, receiver_name)
            changed_email = self.add_sender_name(changed_email, sender_name)
        return changed_email

    def parse_for_subject_verb_object(self, text):
        subject = ""
        activity = ""
        object = ""
        parser = StanfordDependencyParser(
            "C:/Users/Vahid/Documents/Python Scripts/email/stanford_parser/stanford-parser.jar",
            "C:/Users/Vahid/Documents/Python Scripts/email/stanford_parser/stanford-parser-3.7.0-models.jar")
        sentences = parser.raw_parse(text)
        dependencies = sentences.next()
        triples = list(dependencies.triples())
        for set in triples:
            if set[1] == "nsubj":
                # ((u'gave', u'VBD'), u'nsubj', (u'Alice', u'NNP'))
                subject = set[2][0]
            elif set[1] == "iobj":
                # (u'gave', u'VBD'), u'iobj', (u'Bob', u'NNP'))
                object = set[2][0]
            elif set[1] == "dobj":
                # ((u'gave', u'VBD'), u'dobj', (u'pen', u'NN'))
                activity = set[0][0]
        return subject, object, activity

    def remove_numbers_and_unwanted_chars(self, text):
        chars_to_remove = ["1","2","3","4","5","6","7","8","9","0","=","|"]
        changed_text = ""
        for char in text:
            if char in chars_to_remove:
                changed_text = text.replace(char,"")
        return changed_text

    def strip_from_character_occurence(self, text):
        if text.find("<") > 0:
            name = text[:text.find("<")]
        elif text.find("@") > 0:
            name = text[:text.find("@"):]
        else:
            name = text
        return name

    def remove_header_leftovers(self, text):
        name = text
        if text.startswith("From:"):
            name = text.lstrip("From:")
        return name

    def remove_extra_spaces(self, text):
        edited_text = ' '.join(text.split())
        return edited_text

    def clean_names_from_column(self, whole_text):
        print "Original name:",str(whole_text)
        separate_names = whole_text.split(",")
        final_names = []
        final_text = ""
        for text in separate_names:
            if len(text) > 1 and text != 'nan':
                text = text.replace("'", "")
                text = text.lstrip("[")
                text = text.lstrip()
                text = text.rstrip()
                text = text.rstrip("]")
                text = str(text)
                text = text.replace("\\r"," ")
                text = text.replace("\\n", "")
                text = text.replace("'", "")
                text = text.replace("\t"," ")
                changed_text = self.remove_numbers_and_unwanted_chars(text)
                if len(changed_text) > 0:
                    text = changed_text
                else:
                    text = text
                changed_text = self.strip_from_character_occurence(text)
                edited_text = self.remove_header_leftovers(changed_text)
                edited_text = self.remove_extra_spaces(edited_text)
                further_changed_text = self.remove_numbers_and_unwanted_chars(edited_text)
                if len(further_changed_text) > 0:
                    edited_text = further_changed_text.replace(".", " ")
                else:
                    edited_text = edited_text
                    edited_text = edited_text.replace(".", " ")
                final_names.append(edited_text.title())
        final_text = ','.join(final_names)
        final_text = final_text.rstrip(",")
        print "Cleaned name:", final_text
        print "==========="
        return final_text

    def pos_tagging(self):
        print "The function to clean names and putting them in text has been called"
        function_start_time = time.time()
        data = self.data
        pos_tagged_data = pd.DataFrame()
        sub_data = pd.DataFrame()
        message__id_column = []
        thread__id_column = []
        sentence_column = []
        to_names_column = []
        from_names_column = []
        tokenized_sentence = []
        you_i_replaced_column = []
        row_count = 1
        for row in data.iterrows():
            full_email = row[1]["EmailBody"]
            original_from_names = row[1]["FromNames"]
            original_to_names = row[1]["ToNames"]
            print "Cleaning sender name"
            from_names = self.clean_names_from_column(original_from_names)
            print "Cleaning receiver name"
            to_names = self.clean_names_from_column(original_to_names)
            sentences = nltk.sent_tokenize(str(full_email))
            for sentence in sentences:
                message__id_column.append(row[1]["Message-Id"])
                thread__id_column.append(row[1]["Thread-Id"])
                sentence_column.append(sentence)
                # sentence_tokens = nltk.word_tokenize(sentence)
                # tagged_tokens = nltk.pos_tag(sentence_tokens)
                # tokenized_sentence.append(tagged_tokens)
                final_to_name, edited_email, count = self.add_receiver_name(sentence, to_names)
                final_from_name, edited_sentence, count = self.add_sender_name(edited_email, from_names)
                to_names_column.append(final_to_name)
                from_names_column.append(final_from_name)
                you_i_replaced_column.append(edited_sentence)
            print row_count, "row(s) done"
            print "=================="
            row_count = row_count + 1
        loop_end_time = time.time()
        print "all nnames have been cleaned and replaced in text."
        time_taken = loop_end_time - function_start_time
        print "time taken to do that:", time_taken/60
        print "====================================================================="
        pos_tagged_data["Message-Id"] = message__id_column
        pos_tagged_data["Thread-Id"] = thread__id_column
        pos_tagged_data["sentence"] = sentence_column
        pos_tagged_data["ToNames"] = to_names_column
        pos_tagged_data["FromNames"] = from_names_column
        #pos_tagged_data["tagged_sentence"] = tokenized_sentence
        pos_tagged_data["you_i_replaced_text"] = you_i_replaced_column
        return pos_tagged_data

    def tag_text(self):
        eligibility_text = ["i", "you", "we"]
        sentence_wise_data = pd.DataFrame()
        message__id_column = []
        thread__id_column = []
        sentence_column = []
        good_bad_tag_column = []
        subject_column = []
        object_column = []
        activity_column = []
        subject = ""
        object = ""
        activity = ""
        data = self.data
        data = data.head(10)
        row_count = 0
        for row in data.iterrows():
            full_email = row[1]["EmailBody"]
            sentences = sent_tokenize(full_email)
            for sentence in sentences:
                message__id_column.append(row[1]["Message-Id"])
                thread__id_column.append(row[1]["Thread-Id"])
                changed_sentence, count = self.add_sender_name(sentence, row[1]["FromNames"])
                changed_sentence, count = self.add_receiver_name(changed_sentence, row[1]["ToNames"])
                changed_sentence = str(changed_sentence)
                sentence_column.append(changed_sentence)
                tagged_sentence = st.tag(str.lower(changed_sentence).split())
                if any(word in str.lower(changed_sentence).split() for word in eligibility_text):
                    sentences = parser.raw_parse(changed_sentence)
                    dep = sentences.next()
                    '''
                    "Alice gave Bob a pen"
                    ((u'gave', u'VBD'), u'nsubj', (u'Alice', u'NNP'))
                    ((u'gave', u'VBD'), u'iobj', (u'Bob', u'NNP'))
                    ((u'gave', u'VBD'), u'dobj', (u'pen', u'NN'))
                    '''
                    triples = list(dep.triples())
                    for set in triples:
                        if set[1] == "nsubj":
                            subject = set[2][1]
                        elif set[1] == "iobj":
                            object = set[2][1]
                        elif set[1] == "dobj":
                            activity = set[0][0]
                    if any(len(detail) < 1 for detail in [subject, object, activity]):
                        good_bad_tag = "Bad"
                    else:
                        good_bad_tag = "Good"
                    subject_column.append(subject)
                    object_column.append(object)
                    activity_column.append(activity)
                    good_bad_tag_column.append(good_bad_tag)
                elif any(tag[1] == "P" for tag in tagged_sentence):
                    sentences = parser.raw_parse(changed_sentence)
                    dep = sentences.next()
                    '''
                    "Alice gave Bob a pen"
                    ((u'gave', u'VBD'), u'nsubj', (u'Alice', u'NNP'))
                    ((u'gave', u'VBD'), u'iobj', (u'Bob', u'NNP'))
                    ((u'gave', u'VBD'), u'dobj', (u'pen', u'NN'))
                    '''
                    triples = list(dep.triples())
                    for set in triples:
                        if set[1] == "nsubj":
                            subject = set[2][1]
                        elif set[1] == "iobj":
                            object = set[2][1]
                        elif set[1] == "dobj":
                            activity = set[0][0]
                    if any(len(detail) < 1 for detail in [subject, object, activity]):
                        good_bad_tag = "Bad"
                    else:
                        good_bad_tag = "Good"
                    subject_column.append(subject)
                    object_column.append(object)
                    activity_column.append(activity)
                    good_bad_tag_column.append(good_bad_tag)
                else:
                    sentences = parser.raw_parse(changed_sentence)
                    dep = sentences.next()
                    '''
                    "Alice gave Bob a pen"
                    ((u'gave', u'VBD'), u'nsubj', (u'Alice', u'NNP'))
                    ((u'gave', u'VBD'), u'iobj', (u'Bob', u'NNP'))
                    ((u'gave', u'VBD'), u'dobj', (u'pen', u'NN'))
                    '''
                    triples = list(dep.triples())
                    for set in triples:
                        if set[1] == "nsubj":
                            subject = set[2][1]
                        elif set[1] == "iobj":
                            object = set[2][1]
                        elif set[1] == "dobj":
                            activity = set[0][0]
                    if any(len(detail) < 1 for detail in [subject, object, activity]):
                        good_bad_tag = "Bad"
                    else:
                        good_bad_tag = "Good"
                    subject_column.append(subject)
                    object_column.append(object)
                    activity_column.append(activity)
                    good_bad_tag_column.append(good_bad_tag)
            print row_count, "rows done."
            print "====================="
            row_count = row_count + 1
        sentence_wise_data["Message-Id"] = message__id_column
        sentence_wise_data["Thread-Id"] = thread__id_column
        sentence_wise_data["Sentences"] = sentence_column
        sentence_wise_data["Subject"] = subject_column
        sentence_wise_data["Activity"] = activity_column
        sentence_wise_data["Object"] = object_column
        sentence_wise_data["Tag"] = good_bad_tag_column
        print sentence_wise_data.head(10)
        return sentence_wise_data


class notebook_strategy:
    def __init__(self, data_frame):
        print "Notebook Strategy class has been initialized"
        self.data = data_frame

    def parse_st_pos(self, sentence, st):
        results = []
        parsed = st.tag(word_tokenize(sentence))
        pos_map = {}
        for tok in parsed:
            if pos_map.get(tok[1], 0):
                pos_map[tok[1]].append(tok[0])
            else:
                pos_map[tok[1]] = [tok[0]]
        results.append(pos_map)
        return results

    def st_ner_to_iob(self, tagged_sent):
        iob_tagged_sent = []
        prev_tag = "O"
        for token, tag in tagged_sent:
            if tag == "O":  # O
                iob_tagged_sent.append((token, tag))
                prev_tag = tag
                continue
            if tag != "O" and prev_tag == "O":  # Begin NE
                iob_tagged_sent.append((token, "B-" + tag))
                prev_tag = tag
            elif prev_tag != "O" and prev_tag == tag:  # Inside NE
                iob_tagged_sent.append((token, "I-" + tag))
                prev_tag = tag
            elif prev_tag != "O" and prev_tag != tag:  # Adjacent NE
                iob_tagged_sent.append((token, "B-" + tag))
                prev_tag = tag
        return iob_tagged_sent

    def st_iob_joined(self, tagged_sent):
        ner_tagged_sent = self.st_ner_to_iob(tagged_sent)
        results = []
        current_tok = ''
        current_tag = ''
        for token, tag in ner_tagged_sent:
            if tag[:2] == 'B-':
                current_tok = token
                current_tag = tag[2:]
            if tag[:2] == 'I-':
                current_tok += ' ' + token
            if tag[:2] == 'O':
                if len(current_tok) > 0:
                    results.append((current_tok, current_tag))
                    current_tok = ''
                    current_tag = ''
                results.append((token, tag))
        if len(current_tok) > 0:
            results.append((current_tok, current_tag))

        return results

    def parse_st_ner(self, sentence, st):
        results = []
        parsed = st.tag(word_tokenize(sentence))
        parsed = self.st_iob_joined(parsed)
        ner_map = {}
        for tok in parsed:
            if ner_map.get(tok[1], 0):
                ner_map[tok[1]].append(tok[0])
            else:
                ner_map[tok[1]] = [tok[0]]
        results.append(ner_map)
        return results

    def save_tagged_df_to_csv(self, filename, df, pos_tag_set=('NN', 'NNP', 'NNS', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN',
                                                               'VBP', 'VBZ'),
                              ner_tag_set={'PERSON'}):
        f = open(filename, 'w')
        if pos_tag_set is None:
            pos_tag_set = set()
            for i, row in df.iterrows():
                for s in row['st_pos']:
                    for k in s.keys():
                        if k == u'.':
                            continue
                        pos_tag_set.add(k)
            pos_tag_set = sorted(pos_tag_set)

        if ner_tag_set is None:
            ner_tag_set = set()
            for i, row in df.iterrows():
                for s in row['st_ner']:
                    for k in s.keys():
                        if k == u'O':
                            continue
                        ner_tag_set.add(k)
            ner_tag_set = sorted(ner_tag_set)

        fieldnames = ['Message-ID', 'sentence', 'Thread-Id', 'tagged_sentence', 'ToNames', 'FromNames'] + list(ner_tag_set) + list(pos_tag_set)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', delimiter=',', quotechar='"')
        writer.writeheader()
        index = 0
        for i, row in df.iterrows():
            sents = str(row['you_i_replaced_text'])
            for i, pos_tag_row in enumerate(row['st_pos']):
                csvrecord = {'Thread-Id': row['Thread-Id'], 'Message-ID': row['Message-Id'],
                            'ToNames': row['ToNames'], 'FromNames': row['FromNames'], 'sentence': sents,
                             'tagged_sentence': row['st_pos']}
                if len(pos_tag_row) < 1:
                    continue
                for tok in pos_tag_set:
                    if pos_tag_row.get(tok, 0):
                        csvrecord[tok] = ','.join(pos_tag_row[tok])

                ner_tag_row = row['st_ner'][i]
                if len(ner_tag_row) < 1:
                    continue
                for tok in ner_tag_set:
                    if ner_tag_row.get(tok, 0):
                        csvrecord[tok] = ','.join(ner_tag_row[tok])
                writer.writerow(csvrecord)
                index += 1
        f.close()

    def apply_strategy(self):
        print "The function to apply all NLP techniques has been called."
        strategy_start_time = time.time()
        data = self.data
        st_ner = st
        print "Parts of speech tagging has beem initiated."
        data['st_pos'] = data['you_i_replaced_text'].apply(lambda row: self. parse_st_pos(row, st_pos))
        pos_time = time.time()
        print "Parts of Speech tagging on the data has been completed."
        str_time = pos_time - strategy_start_time
        print "Time taken for PoS Tagging to complete:", str_time/60
        print "Names Entity Recognition has been initiated."
        data['st_ner'] = data['you_i_replaced_text'].apply(lambda row: self.parse_st_ner(row, st_ner))
        ner_time = time.time()
        print "Named Entity Recognition Tagging on the data has been completed."
        ner_time = ner_time - strategy_start_time
        print "Time taken for NER Tagging to complete:", ner_time/60
        print "Sample Data:"
        print "==========================================================================="
        print data.head
        print "==========================================================================="
        self.save_tagged_df_to_csv("./final/iter-2_st_ner_st_pos.csv", data)
        data_frame = pd.read_csv("./final/iter-2_st_ner_st_pos.csv", converters={"tagged_sentence": ast.literal_eval})
        self.get_subject_object_data_frame(data_frame)

    def get_subject_object_data_frame(self, data_frame):
        subj_obj_act_start = time.time()
        print "Function to get subject Activity and Object columns from Data has been called."
        final_data_frame = pd.DataFrame()
        subjects = []
        all_nouns = []
        all_verbs = []
        for row in data_frame.iterrows():
            row_nouns = ""
            row_verbs = ""
            all_tags = row[1]['tagged_sentence']
            # [{u'VB': [u'have'], u'VBG': [u'Traveling'], u'NN': [u'business', u'meeting', u'fun', u'trip'], u'.':
            # [u'.']
            # , u'TO': [u'to'], u'IN': [u'out', u'of'], u'VBZ': [u'takes'], u'DT': [u'a', u'the', u'the']}]
            for key in all_tags[0].keys():
                if key.startswith('N'):
                    row_nouns = row_nouns + ", " + ", ".join(all_tags[0][key])
                elif key.startswith('V'):
                    row_verbs = row_verbs + ", " + ", ".join(all_tags[0][key])
            row_nouns = self.objects_minus_person(row_nouns, row[1]['PERSON'])
            subjects.append(self.get_unique_persons(row[1]['PERSON']))
            row_verbs = row_verbs.lstrip(", ")
            all_nouns.append(row_nouns)
            all_verbs.append(row_verbs)
        print "Subject Object Activity Columns have been extracted."
        subj_time = time.time() - subj_obj_act_start
        print "Time Taken to do this:", subj_time/60
        final_data_frame['thread-id'] = data_frame['Thread-Id']
        final_data_frame['message-id'] = data_frame['Message-ID']
        final_data_frame['ToNames'] = data_frame['ToNames']
        final_data_frame['FromNames'] = data_frame['FromNames']
        final_data_frame['Subject'] = subjects
        final_data_frame['Object'] = all_nouns
        final_data_frame['Activity'] = all_verbs
        final_data_frame['sentence'] = data_frame['sentence']
        Good, Bad = self.final_good_bad_email(final_data_frame)
        final_data_frame['Good'] = Good
        final_data_frame['Bad'] = Bad
        final_data_frame.to_csv("./final/iter-2_subject object activity.csv")
        print "The subject activity object dataframe has been created."
        print final_data_frame.head()
        last_data_frame = self.make_count_data_frame(final_data_frame)
        last_data_frame.to_csv("./final/iter-2_last_file.csv")
        print "The last data frame of counts has been created."
        print "====================================================================="

    def objects_minus_person(self, all_nouns, all_names):
        all_nouns = str(all_nouns)
        all_names = str(all_names)
        names_surnames_separate = []
        if len(all_names) > 0:
            if len(all_nouns) > 0 and all_names != "nan" and all_nouns != "nan":
                each_name = all_names.split(",")
                if len(each_name) > 1:  # more than 1 full names were present
                    for full_name in each_name:
                        names = full_name.split(" ")
                        for name in names:
                            names_surnames_separate.append(name)
                else:  # single name
                    names = each_name[0].split(" ")
                    for name in names:
                        names_surnames_separate.append(name)
                each_noun = all_nouns.split(", ")
                final_objects = list(set(each_noun)-set(names_surnames_separate))
                final_objects = ", ".join(final_objects)
                final_objects = final_objects.lstrip(", ")
                print final_objects
            else:
                final_objects = all_nouns
                final_objects = final_objects.lstrip(", ")
                print final_objects
            return final_objects

    def get_unique_persons(self, person_names):
        final_names = person_names
        if str(person_names) != "nan":
            if len(person_names) > 0:
                separate_names = person_names.split(",")
                if len(separate_names) > 1:
                    final_names = " ,".join(list(set(separate_names)))
                else:
                    final_names = person_names
        return final_names

    def final_good_bad_email(self, data_frame):
        print "Giving each sentence Good or Bad tag now."
        good_bad_time = time.time()
        Good = []
        Bad = []
        for row in data_frame.iterrows():
            subject = row[1]['Subject']
            obj = row[1]['Object']
            act = row[1]['Activity']
            if str(subject) != 'nan' and str(obj) != 'nan' and str(act) != 'nan':
                if len(subject) > 0 and len(obj) > 0 and len(act) > 0:
                    Good.append("1")
                    Bad.append(" ")
                else:
                    Good.append(" ")
                    Bad.append("1")
            else:
                Good.append(" ")
                Bad.append("1")
        print "Good Bad tags have been given."
        timess = time.time() - good_bad_time
        print "Time taken to do that:", timess/60
        return Good, Bad

    def make_count_data_frame(self, data_frame):
        print "Making the last count wise dataframe and it's CSV now."
        last_data_frame = pd.DataFrame()
        thread_ids = []
        email_count = []
        bad_count = []
        good_count = []
        unique_threads = pd.unique(data_frame['thread-id'])
        print unique_threads
        for thread in unique_threads:
            relevant_data = data_frame.loc[data_frame['thread-id'] == thread]
            thread_ids.append(thread)
            email_count.append(len(pd.unique(relevant_data['message-id'])))
            bad_data = relevant_data.loc[relevant_data['Bad'] == "1"]
            good_data = relevant_data.loc[relevant_data['Good'] == "1"]
            print "entire data:", len(relevant_data)
            print "bad data:", len(bad_data)
            bad_count.append(len(bad_data))
            print "good data:", len(good_data)
            good_count.append(len(good_data))
            print "====================="
        last_data_frame['thread-id'] = thread_ids
        last_data_frame['Number of Emails'] = email_count
        last_data_frame['Good'] = good_count
        last_data_frame['Bad'] = bad_count
        print "The Data frame of the thread wise good/bad count has been created."
        return last_data_frame

if __name__ == '__main__':
    '''
    data_read_write = ReadWriteData("emails_full_with_thread.csv")
    data_frame_of_csv = data_read_write.read_raw_data()
    data_frame_of_csv = data_frame_of_csv.head(1500)
    pre_process = PreProcessing(data_frame_of_csv)
    thread_wise_emails = pre_process.get_email_by_thread()
    email_quality_tagger = EmailQualityTagger(thread_wise_emails)
    good_emails, bad_emails = email_quality_tagger.tag_by_thread_id()
    edited_dataframe = email_quality_tagger.tag_by_content_and_make_dataframe(data_frame_of_csv, good_emails)
    parser = EmailParser(edited_dataframe)
    new_data = parser.parse_email()
    tagger = ApplyNLPTechiniques("separated_text_ENTIRE_DATA.csv")
    sentence_wise_data = tagger.tag_text()
    sentence_wise_data.to_csv("sentence_wise_data.csv")
    tagger = ApplyNLPTechiniques("separated_text_ENTIRE_DATA.csv")
    pos_tagged = tagger.pos_tagging()
    pos_tagged.to_csv("pos_tagged_ENTIRE_DATA.csv")


    tagger = ApplyNLPTechiniques("pos_tagged_1_5V2.csv")
    df = pd.DataFrame.from_csv("pos_tagged_1_5V2.csv", encoding='utf-8')
    to_name_column = df['ToNames']
    from_name_column = df['FromNames']
    edited_to_name_column = []
    edited_from_name_column = []
    new_replaced_text = []
    for row in df.iterrows():
        print row[0]
        text = str(row[1]['you_i_replaced_text'])
        orig_to_names = str(row[1]['ToNames'])
        orig_from_names = str(row[1]['FromNames'])
        print "TEXT: ", text
        print "ORIGINAL FROM NAME:", orig_from_names
        print "ORIGINAL TO NAME:", orig_to_names
        if orig_to_names != 'nan' and orig_from_names != 'nan':
            print "donon on hain boss"
            edited_to_name, edited_text, count = tagger.add_receiver_name(str(text), orig_to_names)
            edited_from_name, changed_text, count = tagger.add_sender_name(str(edited_text), orig_from_names)
            edited_to_name_column.append(edited_to_name)
            edited_from_name_column.append(edited_from_name)
            new_replaced_text.append(changed_text)
            print "================="
        elif orig_from_names == 'nan' and orig_to_names == 'nan':
            print "both off hain"
            edited_to_name_column.append(orig_to_names)
            edited_from_name_column.append(orig_from_names)
            new_replaced_text.append(text)
        elif orig_from_names == 'nan' and orig_to_names != 'nan':
            print "from name off hai"
            edited_to_name, text_edited, count = tagger.add_receiver_name(text, orig_to_names)
            edited_to_name_column.append(edited_to_name)
            new_replaced_text.append(text_edited)
            edited_from_name_column.append(orig_from_names)
        elif orig_from_names != 'nan' and orig_to_names == 'nan':
            print "to name off hai"
            edited_from_name, text_edited, count = tagger.add_sender_name(text, orig_from_names)
            edited_from_name_column.append(edited_from_name)
            new_replaced_text.append(text_edited)
            edited_to_name_column.append(orig_from_names)
    print len(df['ToNames'])
    print len(df['FromNames'])
    print len(df['Message-Id'])
    print len(edited_from_name_column)
    print len(edited_to_name_column)
    print len(new_replaced_text)
    df['ToNames'] = edited_to_name_column
    df['FromNames'] = edited_from_name_column
    df['you_i_replaced_text'] = new_replaced_text
    df.to_csv("pos_tagged_1_5V3.csv")
    df = pd.read_csv("pos_tagged_1_5V3.csv")
    strategy = notebook_strategy(df.head(1000))
    strategy.apply_strategy()


    df = pd.read_csv("final_tagged_data_replaced_text.csv")
    print len(df)
    df2 = pd.read_csv("pos_tagged_1_5V3.csv")
    print len(df2)
    sub_df = df2.head(1000)
    tagged_sentence = sub_df['tagged_sentence']
    df['tagged_sentence'] = tagged_sentence
    df.to_csv("final_tagged_data_replaced_text.csv")
    '''

    print "==========================================================="
    print "Script started at:", time.ctime()
    print "==========================================================="
    overall_start_time = time.time()
    tagger = ApplyNLPTechiniques("separated_text_ENTIRE_DATA.CSV")
    csv_reading_time = time.time()
    time_taken_by_csv_reading = csv_reading_time - overall_start_time
    print "Data from the CSV file has completely been read"
    print "The time taken by csv reader is:", time_taken_by_csv_reading/60
    data_frame = tagger.pos_tagging()
    strategy = notebook_strategy(data_frame)
    strategy.apply_strategy()
    print "All the work has been done."
    more_time = time.time() - overall_start_time
    print "The time taken to run the entire script is:", more_time/60
    print "==========================================================="
    print "Script ended at:", time.ctime()
    print "==========================================================="
