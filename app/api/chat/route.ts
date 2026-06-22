import { createAnthropic } from "@ai-sdk/anthropic";
import { fromNodeProviderChain } from "@aws-sdk/credential-providers";
import { getTokenProvider } from "@aws/bedrock-token-generator";
import { convertToModelMessages, streamText, UIMessage } from "ai";

const region = process.env.AWS_REGION ?? "us-east-1";

const tokenProvider = getTokenProvider({
  credentials: fromNodeProviderChain(),
  region: region,
});

export async function POST(req: Request) {
  const {
    messages,
    model,
  }: { messages: UIMessage[]; model: string } =
    await req.json();

  const anthropic = createAnthropic({
    baseURL: `https://bedrock-mantle.${region}.api.aws/anthropic/v1`,
    apiKey: await tokenProvider(),
  });

  const result = streamText({
    model: anthropic(model),
    messages: await convertToModelMessages(messages),
    system:
      "You are a helpful assistant that can answer questions and help with tasks",
  });

  return result.toUIMessageStreamResponse({
    sendSources: true,
    sendReasoning: true,
  });
}
