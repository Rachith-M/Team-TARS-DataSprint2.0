"""Rewrite TRAIN rows into casual, English-heavy romanized Hinglish with a local open-weight LLM.

Only train.csv text is sent to the model (a local llama.cpp server); test.csv is never read here.
"""
import argparse
import csv
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEM = ('You rewrite voice-assistant commands. Rewrite the command as casual romanized Hinglish, the way an '
          'urban Indian types to a phone assistant: Latin script only, mix in common English words for apps, devices, '
          'actions and everyday nouns (alarm, set, play, song, cancel, meeting, email, light, volume, weather, please) '
          'where natural. Keep the exact meaning, intent and every name, number, place, date and time. '
          'Output only the rewritten command, nothing else.')
SHOTS = [('मेरे कैलेंडर से कल की बैठक हटा दो', 'kal ki meeting calendar se delete kar do'),
         ('kripaya rediyo par nau sau navaasi chaalu karein', 'please radio pe 989 chalu karo'),
         ('रसोई की बत्तियाँ जला दो', 'kitchen ki lights on kar do')]


def ask(text, url, temperature):
    messages = [{'role': 'user', 'content': SYSTEM + '\n\nCommand: ' + SHOTS[0][0]}, {'role': 'assistant', 'content': SHOTS[0][1]}]
    for source, target in SHOTS[1:]:
        messages += [{'role': 'user', 'content': 'Command: ' + source}, {'role': 'assistant', 'content': target}]
    messages.append({'role': 'user', 'content': 'Command: ' + text})
    body = json.dumps({'messages': messages, 'temperature': temperature, 'max_tokens': 64, 'seed': 7}).encode()
    request = urllib.request.Request(url + '/v1/chat/completions', body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        answer = json.load(response)['choices'][0]['message']['content'].strip().splitlines()
    return answer[0].strip().strip('"') if answer else ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8088')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=.7)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.train.open(encoding='utf-8')))[args.offset:]
    rows = rows[:args.limit] if args.limit else rows
    done = set()
    if args.output.exists():
        done = {json.loads(line)['example_id'] for line in args.output.open(encoding='utf-8')}
    todo = [row for row in rows if row['example_id'] not in done]
    started = time.time()
    with args.output.open('a', encoding='utf-8') as sink, ThreadPoolExecutor(args.workers) as pool:
        for number, (row, text) in enumerate(zip(todo, pool.map(lambda r: ask(r['query_text'], args.url, args.temperature), todo)), 1):
            sink.write(json.dumps({'example_id': row['example_id'], 'source': row['query_text'], 'rewrite': text,
                                   'target': row['target']}, ensure_ascii=False) + '\n')
            sink.flush()
            if number % 50 == 0:
                print(f'{number}/{len(todo)} rows, {number / (time.time() - started):.2f} rows/s', flush=True)
    print('done', len(todo), flush=True)


if __name__ == '__main__':
    main()
