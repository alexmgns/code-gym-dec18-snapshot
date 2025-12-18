import aiohttp
import asyncio

class Fusion:
    def __init__(self, url):
        self.url = url
        self.languages = [
            'python', 'cpp', 'nodejs', 'go', 'go_test', 'java', 'php', 'csharp', 'bash', 'typescript', 'sql',
            'rust', 'cuda', 'lua', 'R', 'perl', 'D_ut', 'ruby', 'scala', 'julia', 'pytest', 'junit',
            'kotlin_script', 'jest', 'verilog', 'python_gpu', 'lean', 'swift', 'racket'
        ]

    async def post(self, url, data):
        print(data)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                print(response)
                if response.status == 200:
                    return await response.json()
                else:
                    print("Error submitting code:", response.status, await response.text())
                    return None

    async def post_code(self, data: dict):
        """
        :param data: dict = {code, language, files}
        :return:
        """
        if data['language'] in self.languages:
            url = f'{self.url}/run_code'
            response = await self.post(url, data)
            return response
        else:
            print(f"Invalid language selected, available languages: {self.languages}")
            return None