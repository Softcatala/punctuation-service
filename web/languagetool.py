#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import language_tool_python

def main():    
    tool = language_tool_python.LanguageTool('ca-ES', remote_server='http://localhost:7001')  # use a remote server API, language Catalan

    with open("60000/riurau-shuf.txt", 'r') as fp:
        lines = fp.readlines()

    with open("review.txt", 'w') as txt_fp, open("review.csv", 'w') as csv_fp:
        line_num = 0
        error = 0
        for line in lines:
            if '\t' in line:
                continue

            line_num += 1
#            if line_num == 100:
#                break 

            matches = tool.check(line)
            if not matches:
                continue

            for idx in range (0, len(matches)):
                match = matches[idx]
                if match.ruleId == "CA_REMOTE_PUNCTUATION_RULE":
                    match.replacements[0] = match.replacements[0].replace(",", "*,*")
                    matches[idx] = match
                    applied = language_tool_python.utils.correct(line, matches)
                    txt_fp.write(f"-- line: {line_num}\n")
                    txt_fp.write(f"{line.strip()}\n" )
                    txt_fp.write(f"{applied.strip()}\n")
                    
                    csv_fp.write(f"{line.strip()}\t{applied.strip()}\n")
                    error += 1
                    break

        print(f"Resum: frases: {line_num}, amb regla de commes: {error}")
                

if __name__ == '__main__':
    main()
