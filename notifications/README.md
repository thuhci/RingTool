# How to slack training bot
The total package of RingTool training experiments usually takes a long time to run. In case of something weird happens during the training, RingTool provides an interface to integrate with Slack. You can set up a Slack bot to send notifications to a specific channel when a training ends. This can help you keep track of the training progress and be alerted in case of any issues. Note that the function is optional and not a must for training. By default the function is disabled and if you want to enable it, just add `--send-notifications-slack` as an argument when running the training script.


## Prerequisites
Firstly make sure you have a [Slack account](https://slack.com/help/articles/218080037-Getting-started-for-new-Slack-users) and a [workspace](https://slack.com/help/articles/206845317-Create-a-Slack-workspace) where you can create a bot. You will also need to install the `slack_sdk` and `toml` package in your Python environment if you haven't.
```sh
pip install slack_sdk toml
```

## Create a Slack Bot
1. Go to the [Slack API page](https://api.slack.com/apps) and click on "Create New App".
2. Choose "From scratch" and provide the app name and workspace.
3. Navigate to the **OAuth & Permissions** section.
4. Under Bot Token Scopes, add the `chat:write` permission to allow the bot to send messages to Slack.
5. Install the app to your workspace by following the provided instructions. You will get an OAuth token (starts with `xoxb-`), which you'll use to authenticate the SDK.
6. If you want the bot to send notifications in a specific channel, make sure the bot is added to that channel. You can do this by inviting the bot to the channel using `/invite @your_bot_name`.

## Fill in the Slack Token in slack_configs.toml
1. Create a file named `slack_configs.toml` in the `configs` directory.
2. Add the following content to the file:
```toml
# slack_configs.toml

[slack]

token = "xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx"

channel = "#general"
```
3. Replace `xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx` with the OAuth token you obtained in the previous step.
4. Replace `#general` with the name of the channel where you want to send notifications. Make sure to include the `#` symbol before the channel name.
5. Save the file.
And that's it! Now, when you run the training script with the `--send-notifications-slack` argument, the bot will send notifications to the specified Slack channel at the end of the training process.

## Examle screenshots
![Success Slack Notifications](screenshots/example-success.jpg)

![Failure Slack Notifications](screenshots/example-failure.png)

![Success Slack Message](screenshots/example-visualization.png)
