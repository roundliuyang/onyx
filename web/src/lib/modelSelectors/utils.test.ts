import type { ModelOptionProvider } from "@/lib/languageModels/options";
import {
  buildModelSelectOptions,
  fromSelectValue,
  toSelectValue,
} from "./utils";

const providers: ModelOptionProvider[] = [
  {
    id: 1,
    name: "OpenAI",
    provider: "openai",
    model_configurations: [
      {
        id: 11,
        name: "gpt-4o",
        is_visible: true,
        max_input_tokens: null,
        supports_image_input: true,
        supports_reasoning: false,
        effectiveDisplayName: "GPT-4o",
      },
      {
        id: 12,
        name: "o3-mini",
        is_visible: false,
        max_input_tokens: null,
        supports_image_input: false,
        supports_reasoning: true,
        effectiveDisplayName: "o3-mini",
      },
      {
        name: "unsaved",
        is_visible: true,
        max_input_tokens: null,
        supports_image_input: false,
        supports_reasoning: false,
        effectiveDisplayName: "Unsaved",
      },
    ],
  },
  {
    id: 2,
    name: null,
    provider: "anthropic",
    model_configurations: [
      {
        id: 21,
        name: "claude-sonnet-4",
        is_visible: true,
        max_input_tokens: null,
        supports_image_input: false,
        supports_reasoning: false,
        effectiveDisplayName: "Claude Sonnet 4",
      },
    ],
  },
];

describe("buildModelSelectOptions", () => {
  test("renders every model given, hidden ones included, under a divider per provider", () => {
    const dividers = buildModelSelectOptions(providers);
    expect(dividers.map((d) => d.title)).toEqual(["Anthropic", "OpenAI"]);
    expect(dividers[1]?.options.map((o) => [o.value, o.title])).toEqual([
      ["11", "GPT-4o"],
      ["12", "o3-mini"],
    ]);
  });

  test("leaves out models without a configuration id", () => {
    const values = buildModelSelectOptions(providers).flatMap((d) =>
      d.options.map((o) => o.value)
    );
    expect(values).not.toContain("unsaved");
    expect(values).toHaveLength(3);
  });

  test("an empty list yields no options", () => {
    expect(buildModelSelectOptions([])).toEqual([]);
  });
});

describe("select value helpers", () => {
  test("round-trip a configuration id", () => {
    expect(toSelectValue(11)).toBe("11");
    expect(toSelectValue(null)).toBe("");
    expect(fromSelectValue("11")).toBe(11);
    expect(fromSelectValue("")).toBeNull();
    expect(fromSelectValue("nope")).toBeNull();
  });
});
