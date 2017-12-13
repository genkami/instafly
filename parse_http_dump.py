"""
mitmdumpの出力から、instagramとの通信を抜き出し、
ユーザー情報と画像を保存する
"""

import re
import json
import csv
import io
import argparse
import os
import shutil
import mitmproxy.io

class Media(object):
    def __init__(self, id, owner, likes, thumbnails):
        self.id = id
        self.owner = owner
        self.likes = likes
        self.thumbnails = thumbnails

    @classmethod
    def from_json(_, js):
        return Media(
            id=js['id'],
            owner=js['owner']['id'],
            likes=js['likes']['count'],
            thumbnails=list(map(lambda r: r['src'], js['thumbnail_resources']))
        )

class User(object):
    def __init__(self, id, name, followed_by, media):
        self.id = id
        self.name = name
        self.followed_by = followed_by
        self.media = media

    @classmethod
    def from_json(_, js):
        return User(
            id=js['user']['id'],
            name=js['user']['full_name'],
            followed_by=js['user']['followed_by']['count'],
            media=list(map(Media.from_json, js['user']['media']['nodes']))
        )

class FlowHandler(object):
    def can_accept(self, flow):
        raise Exception('cannot call abstract method')

    def handle(self, flow):
        raise Exception('cannot call abstract method')

    def handle_stream(self, stream):
        for flow in stream:
            if self.can_accept(flow):
                self.handle(flow)

    def handle_stream_file(self, path):
        with open(path, 'rb') as f:
            flow_reader = mitmproxy.io.FlowReader(f)
            self.handle_stream(flow_reader.stream())

class UserHandler(FlowHandler):
    ACCEPTABLE_HOST = 'www.instagram.com'
    ACCEPTABLE_PATH_PAT = re.compile(r'/[a-zA-Z0-9._]+/\?__a=1')

    def __init__(self):
        self.users = {}
        self.thumbnails = {}


    def can_accept(self, flow):
        return flow.request.host == self.ACCEPTABLE_HOST and \
            self.ACCEPTABLE_PATH_PAT.match(flow.request.path) is not None

    def handle(self, flow):
        user = User.from_json(json.loads(flow.response.text))
        self.users[user.id] = user
        for media in user.media:
            for thumb in media.thumbnails:
                self.thumbnails[thumb] = media

class Image(object):
    def __init__(self, id, likes, content, owner):
        self.id = id
        self.likes = likes
        self.content = content
        self.owner = owner

class MediaThumbHandler(FlowHandler):
    ACCEPTABLE_HOST_PAT = re.compile(r'.*cdninstagram.com')

    def __init__(self, users, thumbnails):
        self.users = users
        self.thumbnails = thumbnails
        self.images = {}

    def can_accept(self, flow):
        return self.ACCEPTABLE_HOST_PAT.match(flow.request.host) is not None

    def handle(self, flow):
        url = 'https://' + flow.request.host + flow.request.path
        if url in self.thumbnails:
            media = self.thumbnails[url]
            self.images[media.id] = Image(
                id=media.id, likes=media.likes,
                owner=media.owner, content=flow.response.content
            )

class CompoundHandler(FlowHandler):
    def __init__(self, *handlers):
        self.__handlers = handlers

    def can_accept(self, flow):
        return any(map(lambda h: h.can_accept(flow), self.__handlers))

    def handle(self, flow):
        for h in self.__handlers:
            if h.can_accept(flow):
                h.handle(flow)

def extract_instagram_data_from_dump_file(
        dump_file_path, user_file_path, media_file_path, image_dir_path):
    user_handler = UserHandler()
    media_handler = MediaThumbHandler(user_handler.users, user_handler.thumbnails)
    handler = CompoundHandler(user_handler, media_handler)
    handler.handle_stream_file(dump_file_path)

    image_paths = save_imgs(media_handler.images, image_dir_path)
    save_media_data(media_handler.images, image_paths, media_file_path)

    save_user_data(user_handler.users, user_file_path)

# TODO: 画像データはjpgが前提になっているので、必要であれば直す
def save_imgs(images, image_dir_path):
    image_paths = {}
    for img in images.values():
        outpath = os.path.join(image_dir_path, '{}.jpg'.format(img.id))
        image_paths[img.id] = outpath
        with open(outpath, 'wb') as fout:
            fin = io.BytesIO(img.content)
            shutil.copyfileobj(fin, fout)
    return image_paths

def save_media_data(images, image_paths, media_file_path):
    with open(media_file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow('id likes owner path'.split())
        for img in images.values():
            path = image_paths.get(img.id, "")
            writer.writerow([img.id, img.likes, img.owner, path])

def save_user_data(users, user_file_path):
    with open(user_file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow('id name followed_by'.split())
        for user in users.values():
            writer.writerow([user.id, user.name, user.followed_by])

def main():
    argparser = argparse.ArgumentParser(
        description='mitmdumpの出力からインスタの画像を収集する'
    )
    argparser.add_argument('-f', metavar='MITM_DUMP_FILE_PATH', required=True,
                           help='mitmdumpの出力ファイル')
    argparser.add_argument('-u', metavar='USER_FILE_PATH', required=True,
                           help='収集したユーザーの情報を保存するcsvファイルのパス')
    argparser.add_argument('-m', metavar='MEDIA_FILE_PATH', required=True,
                           help='収集した投稿の情報を保存するcsvファイルのパス')
    argparser.add_argument('-d', metavar='IMAGE_DIR_PATH', required=True,
                           help='収集した画像を保存するディレクトリのパス')
    args = argparser.parse_args()
    extract_instagram_data_from_dump_file(
        dump_file_path=args.f,
        user_file_path=args.u,
        media_file_path=args.m,
        image_dir_path=args.d
    )

if __name__ == '__main__':
    main()
